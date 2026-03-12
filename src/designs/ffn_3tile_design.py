# FFN (Feed-Forward Network) Block: MatMul_up → GELU → MatMul_down (3-tile pipeline)
#
# Tile 1: GEMM  — X[seq,hd] × W_up[hd,ffn] → intermediate[seq,ffn]
# Tile 2: GELU  — elementwise GELU activation
# Tile 3: GEMM  — intermediate[seq,ffn] × W_down[ffn,hd] → output[seq,hd]
#
# Data flows tile-to-tile. Intermediate activations NEVER return to host.
#
# Processes m=4 rows per block, loops seq/m times.
# W_up is sent once to tile 1. W_down is sent once to tile 3.
#
# Runtime args:
#   arg0 = X[seq*hd] + W_up[hd*ffn] packed
#   arg1 = W_down[ffn*hd]
#   arg2 = output[seq*hd]

import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def ffn_3tile(seq=128, hd=64, ffn=256, m=4):
    """
    FFN block: X × W_up → GELU → × W_down → output

    seq: sequence length (128)
    hd:  hidden dimension (64 for proof-of-concept, 512 for Kokoro)
    ffn: FFN intermediate dimension (256 for PoC, 2048 for Kokoro)
    m:   rows per block (must divide seq, ≥4 for mmul)
    """
    assert seq % m == 0
    n_blocks = seq // m  # 32

    dev = NPU2()

    # Types
    x_block_ty = np.ndarray[(m * hd,), np.dtype[bfloat16]]           # 4×64 = 256
    w_up_ty = np.ndarray[(hd * ffn,), np.dtype[bfloat16]]            # 64×256 = 16384
    inter_block_ty = np.ndarray[(m * ffn,), np.dtype[bfloat16]]      # 4×256 = 1024
    w_down_ty = np.ndarray[(ffn * hd,), np.dtype[bfloat16]]          # 256×64 = 16384
    out_block_ty = np.ndarray[(m * hd,), np.dtype[bfloat16]]         # 4×64 = 256

    input_ty = np.ndarray[(seq * hd + hd * ffn,), np.dtype[bfloat16]]
    w_down_full_ty = np.ndarray[(ffn * hd,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(seq * hd,), np.dtype[bfloat16]]

    # Kernels
    # Tile 1: FFN up matmul (M=4, K=hd, N=ffn)
    up_matmul = Kernel(
        "matmul_ffn_up", "ffn_up_matmul.o", [x_block_ty, w_up_ty, inter_block_ty]
    )
    up_zero = Kernel("zero_ffn_up", "ffn_up_matmul.o", [inter_block_ty])

    # Tile 2: GELU (processes m*ffn = 1024 elements per call)
    gelu_kernel = Kernel(
        "gelu_bf16", "gelu_ffn.o", [inter_block_ty, inter_block_ty]
    )

    # Tile 3: FFN down matmul (M=4, K=ffn, N=hd)
    down_matmul = Kernel(
        "matmul_ffn_down", "ffn_down_matmul.o", [inter_block_ty, w_down_ty, out_block_ty]
    )
    down_zero = Kernel("zero_ffn_down", "ffn_down_matmul.o", [out_block_ty])

    # FIFOs
    of_x = ObjectFifo(x_block_ty, name="x_in", depth=2)           # host → tile 1
    of_w_up = ObjectFifo(w_up_ty, name="w_up_in", depth=1)        # host → tile 1
    of_inter1 = ObjectFifo(inter_block_ty, name="inter_t2t", depth=2)  # tile 1 → tile 2
    of_inter2 = ObjectFifo(inter_block_ty, name="gelu_t2t", depth=2)   # tile 2 → tile 3
    of_w_down = ObjectFifo(w_down_ty, name="w_down_in", depth=1)  # host → tile 3
    of_out = ObjectFifo(out_block_ty, name="out", depth=2)         # tile 3 → host

    # Worker 1: FFN Up MatMul
    def up_body(x_fifo, w_fifo, inter_fifo, zero_fn, matmul_fn):
        elem_w = w_fifo.acquire(1)
        for _ in range_(n_blocks):
            elem_x = x_fifo.acquire(1)
            elem_inter = inter_fifo.acquire(1)
            zero_fn(elem_inter)
            matmul_fn(elem_x, elem_w, elem_inter)
            x_fifo.release(1)
            inter_fifo.release(1)
        w_fifo.release(1)

    w_up_worker = Worker(
        up_body,
        fn_args=[of_x.cons(), of_w_up.cons(), of_inter1.prod(),
                 up_zero, up_matmul],
    )

    # Worker 2: GELU
    def gelu_body(in_fifo, out_fifo, gelu_fn):
        for _ in range_(n_blocks):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            gelu_fn(elem_in, elem_out)
            in_fifo.release(1)
            out_fifo.release(1)

    w_gelu = Worker(
        gelu_body,
        fn_args=[of_inter1.cons(), of_inter2.prod(), gelu_kernel],
    )

    # Worker 3: FFN Down MatMul
    def down_body(inter_fifo, w_fifo, out_fifo, zero_fn, matmul_fn):
        elem_w = w_fifo.acquire(1)
        for _ in range_(n_blocks):
            elem_inter = inter_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            zero_fn(elem_out)
            matmul_fn(elem_inter, elem_w, elem_out)
            inter_fifo.release(1)
            out_fifo.release(1)
        w_fifo.release(1)

    w_down = Worker(
        down_body,
        fn_args=[of_inter2.cons(), of_w_down.cons(), of_out.prod(),
                 down_zero, down_matmul],
    )

    rt = Runtime()
    with rt.sequence(input_ty, w_down_full_ty, output_ty) as (X_Wup_in, Wdown_in, Out):
        rt.start(w_up_worker)
        rt.start(w_gelu)
        rt.start(w_down)

        # Fill X blocks from first portion of arg0
        tap_x = TensorAccessPattern(
            tensor_dims=[seq * hd + hd * ffn],
            offset=0,
            sizes=[1, n_blocks, m, hd],
            strides=[0, m * hd, hd, 1],
        )
        rt.fill(of_x.prod(), X_Wup_in, tap=tap_x)

        # Fill W_up from second portion of arg0 (sent once)
        tap_w_up = TensorAccessPattern(
            tensor_dims=[seq * hd + hd * ffn],
            offset=seq * hd,
            sizes=[1, 1, hd, ffn],
            strides=[0, 0, ffn, 1],
        )
        rt.fill(of_w_up.prod(), X_Wup_in, tap=tap_w_up)

        # Fill W_down (sent once, arg1)
        rt.fill(of_w_down.prod(), Wdown_in)

        # Drain output
        rt.drain(of_out.cons(), Out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--hd", type=int, default=64)
    p.add_argument("--ffn", type=int, default=256)
    p.add_argument("--m", type=int, default=4)
    args = p.parse_args()

    module = ffn_3tile(args.seq, args.hd, args.ffn, args.m)
    print(module)
