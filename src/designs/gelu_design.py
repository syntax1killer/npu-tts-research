# GELU activation — single tile, processes N elements in chunks
#
# Input:  x[N] bf16
# Output: y[N] bf16
#
# Processes TILE_SIZE elements per kernel call, loops N/TILE_SIZE times.
# TILE_SIZE must match GELU_SIZE compile-time define in the kernel.
#
# Runtime sequence:
#   arg0 = x_in  [N]
#   arg1 = unused (placeholder for 3-arg sequence)
#   arg2 = x_out [N]

import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def gelu_design(N=32768, TILE_SIZE=256):
    assert N % TILE_SIZE == 0
    n_iters = N // TILE_SIZE

    dev = NPU2()

    tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    gelu_kernel = Kernel(
        "gelu_bf16", "gelu.o", [tile_ty, tile_ty]
    )

    of_in = ObjectFifo(tile_ty, name="x_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="x_out", depth=2)

    def worker_body(in_fifo, out_fifo, gelu_fn):
        for _ in range_(n_iters):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            gelu_fn(elem_in, elem_out)
            in_fifo.release(1)
            out_fifo.release(1)

    w = Worker(
        worker_body,
        fn_args=[
            of_in.cons(),
            of_out.prod(),
            gelu_kernel,
        ],
    )

    rt = Runtime()
    with rt.sequence(full_ty, full_ty, full_ty) as (X_in, _unused, X_out):
        rt.start(w)
        rt.fill(of_in.prod(), X_in)
        rt.drain(of_out.cons(), X_out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32768)
    p.add_argument("--tile", type=int, default=256)
    args = p.parse_args()

    module = gelu_design(args.n, args.tile)
    print(module)
