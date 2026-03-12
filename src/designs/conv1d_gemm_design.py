# Conv1d as GEMM — multi-tile design
#
# Uses 4 tiles (4 rows × 1 col) to compute:
#   A[M, K] × B[K, N] = C[M, N]
# where M = L_out, K = C_in * kernel_size, N = C_out
#
# Each tile computes n_col_iters (m, n) output blocks.
# K-reduction happens within each block (loop over K/k tiles).

import numpy as np
import argparse
import sys
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def conv1d_gemm(M=128, K=64, N=32, m=32, k=64, n=32):
    """
    GEMM for Conv1d: A[M,K] × B[K,N] = C[M,N]

    Each of 4 tiles processes one (m, N) row strip of C.
    Within each strip, iterates over N/n column blocks.
    For each (m, n) output block, loops K/k times for K-reduction.
    """
    assert M % m == 0, f"M={M} must be divisible by m={m}"
    assert K % k == 0, f"K={K} must be divisible by k={k}"
    assert N % n == 0, f"N={N} must be divisible by n={n}"

    n_rows = M // m
    k_iters = K // k
    n_col_iters = N // n

    dev = NPU2()

    # Types
    A_tile_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    B_tile_ty = np.ndarray[(k * n,), np.dtype[bfloat16]]
    C_tile_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    A_full_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_full_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    C_full_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    # Kernels
    matmul_kernel = Kernel(
        "matmul_bf16_bf16", "conv1d.o", [A_tile_ty, B_tile_ty, C_tile_ty]
    )
    zero_kernel = Kernel("zero_bf16", "conv1d.o", [C_tile_ty])

    # ObjectFIFOs — one set per tile row
    of_A = [
        ObjectFifo(A_tile_ty, name=f"A_{i}", depth=2) for i in range(n_rows)
    ]
    of_B = [
        ObjectFifo(B_tile_ty, name=f"B_{i}", depth=2) for i in range(n_rows)
    ]
    of_C = [
        ObjectFifo(C_tile_ty, name=f"C_{i}", depth=2) for i in range(n_rows)
    ]

    # Workers
    workers = []
    for i in range(n_rows):
        def make_body(ki=k_iters, nci=n_col_iters):
            def body(a_fifo, b_fifo, c_fifo, zero_fn, matmul_fn):
                # Iterate over N/n column blocks
                for _ in range_(nci):
                    elem_c = c_fifo.acquire(1)
                    zero_fn(elem_c)

                    # K-reduction loop
                    for _ in range_(ki):
                        elem_a = a_fifo.acquire(1)
                        elem_b = b_fifo.acquire(1)
                        matmul_fn(elem_a, elem_b, elem_c)
                        a_fifo.release(1)
                        b_fifo.release(1)

                    c_fifo.release(1)

            return body

        w = Worker(
            make_body(),
            fn_args=[
                of_A[i].cons(),
                of_B[i].cons(),
                of_C[i].prod(),
                zero_kernel,
                matmul_kernel,
            ],
        )
        workers.append(w)

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(A_full_ty, B_full_ty, C_full_ty) as (A, B, C):
        for w in workers:
            rt.start(w)

        # Fill A — each tile gets n_col_iters * k_iters tiles of (m, k)
        # For each column block, re-send the same A rows (different B columns)
        for i in range(n_rows):
            tap = TensorAccessPattern(
                tensor_dims=[M * K],
                offset=i * m * K,
                sizes=[n_col_iters, k_iters, m, k],
                strides=[0, k, K, 1],
            )
            rt.fill(of_A[i].prod(), A, tap=tap)

        # Fill B — k_iters tiles of (k, n) for each column block
        # Column block j starts at offset j*n in B
        for i in range(n_rows):
            tap = TensorAccessPattern(
                tensor_dims=[K * N],
                offset=0,
                sizes=[n_col_iters, k_iters, k, n],
                strides=[n, k * N, N, 1],
            )
            rt.fill(of_B[i].prod(), B, tap=tap)

        # Drain C — each tile produces n_col_iters output blocks
        # C[i*m:(i+1)*m, j*n:(j+1)*n] for j in 0..n_col_iters-1
        for i in range(n_rows):
            tap = TensorAccessPattern(
                tensor_dims=[M * N],
                offset=i * m * N,
                sizes=[1, n_col_iters, m, n],
                strides=[0, n, N, 1],
            )
            rt.drain(of_C[i].cons(), C, tap=tap, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=32)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    args = p.parse_args()

    module = conv1d_gemm(args.M, args.K, args.N, args.m, args.k, args.n)
    print(module)
