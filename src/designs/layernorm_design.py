# LayerNorm operator — single tile, processes N_ROWS rows of DIM elements
#
# Input:  x[N_ROWS, DIM] bf16
# Params: [gamma(DIM) | beta(DIM)] bf16  (packed into one buffer)
# Output: y[N_ROWS, DIM] bf16
#
# The tile processes one row at a time, looping N_ROWS times.
# Params (gamma+beta) are sent once and reused across all rows.
#
# Uses 2 input DMA channels (x_in, params) + 1 output (x_out) = within limits.
#
# Runtime sequence:
#   arg0 = x_in  [N_ROWS * DIM]
#   arg1 = params [2 * DIM]
#   arg2 = x_out [N_ROWS * DIM]

import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def layernorm_design(N_ROWS=128, DIM=256):
    dev = NPU2()

    row_ty = np.ndarray[(DIM,), np.dtype[bfloat16]]
    full_ty = np.ndarray[(N_ROWS * DIM,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(2 * DIM,), np.dtype[bfloat16]]  # gamma + beta packed

    ln_kernel = Kernel(
        "layernorm_bf16", "layernorm.o", [row_ty, params_ty, row_ty]
    )

    # ObjectFIFOs — 2 input + 1 output (within DMA channel limits)
    of_in = ObjectFifo(row_ty, name="x_in", depth=2)
    of_params = ObjectFifo(params_ty, name="params", depth=1)
    of_out = ObjectFifo(row_ty, name="x_out", depth=2)

    def worker_body(in_fifo, params_fifo, out_fifo, ln_fn):
        elem_params = params_fifo.acquire(1)
        for _ in range_(N_ROWS):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            ln_fn(elem_in, elem_params, elem_out)
            in_fifo.release(1)
            out_fifo.release(1)
        params_fifo.release(1)

    w = Worker(
        worker_body,
        fn_args=[
            of_in.cons(),
            of_params.cons(),
            of_out.prod(),
            ln_kernel,
        ],
    )

    rt = Runtime()
    with rt.sequence(full_ty, params_ty, full_ty) as (X_in, Params, X_out):
        rt.start(w)

        # Fill input rows
        rt.fill(of_in.prod(), X_in)

        # Fill params (gamma + beta packed, sent once)
        rt.fill(of_params.prod(), Params)

        # Drain output
        rt.drain(of_out.cons(), X_out, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    args = p.parse_args()

    module = layernorm_design(args.rows, args.dim)
    print(module)
