# MHA Single-Head Attention: QK^T → Softmax → AV (3-tile chained pipeline)
#
# Tile 1: GEMM — Q[seq,hd] × K^T[hd,seq] → attn[seq,seq]
# Tile 2: Softmax — row-wise softmax on attention scores
# Tile 3: GEMM — attn[seq,seq] × V[seq,hd] → out[seq,hd]
#
# Data flows tile-to-tile. Attention scores NEVER return to host memory.
# This is the core MHA computation for one attention head.
#
# Processes m=4 rows per block, loops seq/m times.
# K^T is sent once to tile 1. V is sent once to tile 3.
#
# Runtime args:
#   arg0 = Q[seq*hd] + K^T[hd*seq] packed
#   arg1 = V[seq*hd]
#   arg2 = output[seq*hd]

import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def mha_attention_3tile(seq=128, hd=64, m=4):
    """
    Full single-head attention: Q×K^T → softmax → ×V → output

    seq: sequence length (128 for Kokoro)
    hd:  head dimension (64 for Kokoro)
    m:   rows per block (must divide seq, ≥4 for mmul)
    """
    assert seq % m == 0
    n_blocks = seq // m  # 32

    dev = NPU2()

    # Types
    q_block_ty = np.ndarray[(m * hd,), np.dtype[bfloat16]]      # 4×64 = 256
    kt_ty = np.ndarray[(hd * seq,), np.dtype[bfloat16]]         # 64×128 = 8192
    attn_block_ty = np.ndarray[(m * seq,), np.dtype[bfloat16]]   # 4×128 = 512
    v_ty = np.ndarray[(seq * hd,), np.dtype[bfloat16]]          # 128×64 = 8192
    out_block_ty = np.ndarray[(m * hd,), np.dtype[bfloat16]]    # 4×64 = 256

    input_ty = np.ndarray[(seq * hd + hd * seq,), np.dtype[bfloat16]]
    v_full_ty = np.ndarray[(seq * hd,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(seq * hd,), np.dtype[bfloat16]]

    # Kernels
    # Tile 1: QKT matmul (M=4, K=64, N=128)
    qkt_matmul = Kernel(
        "matmul_bf16_bf16", "qkt_matmul.o", [q_block_ty, kt_ty, attn_block_ty]
    )
    qkt_zero = Kernel("zero_bf16", "qkt_matmul.o", [attn_block_ty])

    # Tile 2: Multi-row softmax (4 rows of 128)
    softmax_kernel = Kernel(
        "softmax_multirow_bf16", "softmax_mr.o", [attn_block_ty, attn_block_ty]
    )

    # Tile 3: AV matmul (M=4, K=128, N=64) — renamed to avoid symbol clash
    av_matmul = Kernel(
        "matmul_av", "av_matmul.o", [attn_block_ty, v_ty, out_block_ty]
    )
    av_zero = Kernel("zero_av", "av_matmul.o", [out_block_ty])

    # FIFOs
    of_q = ObjectFifo(q_block_ty, name="q_in", depth=2)       # host → tile 1
    of_kt = ObjectFifo(kt_ty, name="kt_in", depth=1)          # host → tile 1
    of_attn = ObjectFifo(attn_block_ty, name="attn_t2t", depth=2)  # tile 1 → tile 2
    of_sm = ObjectFifo(attn_block_ty, name="sm_t2t", depth=2)      # tile 2 → tile 3
    of_v = ObjectFifo(v_ty, name="v_in", depth=1)              # host → tile 3
    of_out = ObjectFifo(out_block_ty, name="out", depth=2)     # tile 3 → host

    # Worker 1: QK^T GEMM
    def qkt_body(q_fifo, kt_fifo, attn_fifo, zero_fn, matmul_fn):
        elem_kt = kt_fifo.acquire(1)
        for _ in range_(n_blocks):
            elem_q = q_fifo.acquire(1)
            elem_attn = attn_fifo.acquire(1)
            zero_fn(elem_attn)
            matmul_fn(elem_q, elem_kt, elem_attn)
            q_fifo.release(1)
            attn_fifo.release(1)
        kt_fifo.release(1)

    w_qkt = Worker(
        qkt_body,
        fn_args=[of_q.cons(), of_kt.cons(), of_attn.prod(),
                 qkt_zero, qkt_matmul],
    )

    # Worker 2: Softmax
    def softmax_body(attn_fifo, sm_fifo, sm_fn):
        for _ in range_(n_blocks):
            elem_in = attn_fifo.acquire(1)
            elem_out = sm_fifo.acquire(1)
            sm_fn(elem_in, elem_out)
            attn_fifo.release(1)
            sm_fifo.release(1)

    w_sm = Worker(
        softmax_body,
        fn_args=[of_attn.cons(), of_sm.prod(), softmax_kernel],
    )

    # Worker 3: AV GEMM
    def av_body(sm_fifo, v_fifo, out_fifo, zero_fn, matmul_fn):
        elem_v = v_fifo.acquire(1)
        for _ in range_(n_blocks):
            elem_attn = sm_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            zero_fn(elem_out)
            matmul_fn(elem_attn, elem_v, elem_out)
            sm_fifo.release(1)
            out_fifo.release(1)
        v_fifo.release(1)

    w_av = Worker(
        av_body,
        fn_args=[of_sm.cons(), of_v.cons(), of_out.prod(),
                 av_zero, av_matmul],
    )

    rt = Runtime()
    with rt.sequence(input_ty, v_full_ty, output_ty) as (QKT_in, V_in, Out):
        rt.start(w_qkt)
        rt.start(w_sm)
        rt.start(w_av)

        # Fill Q blocks from first portion of arg0
        tap_q = TensorAccessPattern(
            tensor_dims=[seq * hd + hd * seq],
            offset=0,
            sizes=[1, n_blocks, m, hd],
            strides=[0, m * hd, hd, 1],
        )
        rt.fill(of_q.prod(), QKT_in, tap=tap_q)

        # Fill K^T from second portion of arg0 (sent once)
        tap_kt = TensorAccessPattern(
            tensor_dims=[seq * hd + hd * seq],
            offset=seq * hd,
            sizes=[1, 1, hd, seq],
            strides=[0, 0, seq, 1],
        )
        rt.fill(of_kt.prod(), QKT_in, tap=tap_kt)

        # Fill V (sent once, arg1)
        rt.fill(of_v.prod(), V_in)

        # Drain output
        rt.drain(of_out.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--hd", type=int, default=64)
    p.add_argument("--m", type=int, default=4)
    args = p.parse_args()

    module = mha_attention_3tile(args.seq, args.hd, args.m)
    print(module)
