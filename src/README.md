# Source Code

This directory contains the complete source code for the bare-metal NPU inference pipeline described in the paper. The code targets AMD AIE2P (XDNA2, Ryzen AI) via MLIR-AIE and IRON.

## Directory Structure

### `kernels/` — AIE Kernel C++ (runs on NPU tiles)

Compiled with Peano/clang++ targeting `aie2p-none-unknown-elf`. Each kernel processes one chunk of data per invocation; the IRON design scripts handle tiling and DMA.

| File | Operation | Notes |
|------|-----------|-------|
| `layernorm_kernel.cpp` | Layer Normalization | 2-pass: mean/var → normalize. Uses `aie::vector<bfloat16, 16>` |
| `gelu_kernel.cpp` | GELU Activation | Tanh approximation via `aie::tanh` intrinsic |
| `softmax_kernel.cpp` | Softmax | 3-pass: max → exp2 → normalize. Uses `aie::exp2` |
| `conv1d_kernel.cpp` | GEMM (General MatMul) | Uses `aie::mmul<4,8,8>` hardware intrinsic. Configurable M/K/N via defines |
| `mha_kernel.cpp` | Multi-Head Attention | Auto-vectorized loops (explicit `aie::mmul` caused legalization failures) |
| `add_kernel.cpp` | Element-wise Add | Residual connections. Simple vector add |

**Key patterns:**
- All kernels use `event0()`/`event1()` for cycle counting
- Scalar BF16 ops are broken on AIE2P — must use `aie::vector` ops
- `aie::mmul<4,8,8, bfloat16, bfloat16>` is the correct shape for AIE2P

### `designs/` — MLIR-AIE / IRON Design Scripts (generates xclbin)

Python scripts using the IRON API to describe multi-tile compute graphs. Each generates MLIR IR which is compiled via `aiecc.py` into `.xclbin` (NPU executable) + `insts.bin` (instruction stream).

| File | Operator | Tiles | Key Pattern |
|------|----------|-------|-------------|
| `layernorm_design.py` | LayerNorm | 1 | Row-by-row loop, params sent once |
| `gelu_design.py` | GELU | 1 | Chunked elementwise processing |
| `softmax_design.py` | Softmax | 1 | Row-by-row softmax |
| `mha_3tile_design.py` | Full Attention Head | 3 (chained) | Q*K^T → Softmax → Attn*V, tile-to-tile dataflow |
| `ffn_3tile_design.py` | FFN Block | 3 (chained) | MatMul_up → GELU → MatMul_down, tile-to-tile |
| `conv1d_gemm_design.py` | Multi-tile GEMM | 4 (parallel) | K-reduction loop, TensorAccessPattern striding |

**Key concepts:**
- `ObjectFifo`: Producer-consumer buffers between tiles (or between host DMA and tiles)
- `TensorAccessPattern`: Maps flat host buffers to tiled DMA transfers without data copying
- `Worker`: A function that runs on one AIE tile, acquiring/releasing FIFO elements
- `range_()`: IRON's compile-time loop (generates MLIR `scf.for`)

### `host/` — Windows Host Code (XRT submission + benchmarks)

C++ code compiled with MSVC (`cl.exe /std:c++20`) that allocates XRT buffer objects, uploads data, submits kernels to NPU, and measures performance.

| File | Purpose | Key Feature |
|------|---------|-------------|
| `benchmark_iron_gemm.cpp` | IRON GEMM benchmark | Warmup + timed iterations, GFLOPS/bandwidth calculation, CPU verification |
| `albert_bench.cpp` | Full ALBERT pass benchmark | Hybrid CPU/NPU, real ONNX weights, per-op timing, AVX2 MHA, precision dump |

`albert_bench.cpp` is the main measurement tool for the paper's results. It supports:
- `--cpu-mha`: FP32 AVX2 multi-head attention on CPU (for precision isolation)
- `--iron-mha`: IRON MHA on NPU
- `--weights-dir`: Load real weights extracted from ONNX model
- `--dump-dir`: Save per-pass outputs for precision analysis against FP32 reference

## Build Requirements

- **WSL2**: MLIR-AIE + IRON + Peano (for kernel compilation and design script execution)
- **Windows**: MSVC + XRT SDK (for host code compilation)
- **Hardware**: AMD Ryzen AI with XDNA2 NPU (tested on Ryzen AI 9 HX 370)

See [pipeline/build_flow.md](../pipeline/build_flow.md) for the full WSL-to-Windows-to-NPU build pipeline.

## Note on IRON GEMM and MHA

The production GEMM and MHA operators used for the paper's benchmark results come from AMD's [IRON](https://github.com/amd/IRON) repository (`iron/operators/gemm/` and `iron/operators/mha/`). These are not included here as they are part of IRON's codebase. Our custom kernels and designs (in this directory) were used for development and early benchmarking; the IRON operators were used for the final performance numbers reported in the paper.
