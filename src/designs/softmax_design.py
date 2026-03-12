# Softmax operator — single tile, processes N_ROWS rows of DIM elements
#
# Input:  x[N_ROWS * DIM] bf16
# Output: y[N_ROWS * DIM] bf16
#
# Each row is softmaxed independently.
# DIM must match SM_DIM kernel compile-time define.

import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def softmax_design(N_ROWS=128, DIM=128):
    dev = NPU2()

    row_ty = np.ndarray[(DIM,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(N_ROWS * DIM,), np.dtype[bfloat16]]

    sm_kernel = Kernel("softmax_bf16", "softmax.o", [row_ty, row_ty])

    of_in = ObjectFifo(row_ty, name="x_in", depth=2)
    of_out = ObjectFifo(row_ty, name="x_out", depth=2)

    def worker_body(in_fifo, out_fifo, sm_fn):
        for _ in range_(N_ROWS):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            sm_fn(elem_in, elem_out)
            in_fifo.release(1)
            out_fifo.release(1)

    w = Worker(
        worker_body,
        fn_args=[of_in.cons(), of_out.prod(), sm_kernel],
    )

    rt = Runtime()
    with rt.sequence(full_ty, full_ty, full_ty) as (X_in, _unused, X_out):
        rt.start(w)
        rt.fill(of_in.prod(), X_in)
        rt.drain(of_out.cons(), X_out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()

    module = softmax_design(args.rows, args.dim)
    print(module)
