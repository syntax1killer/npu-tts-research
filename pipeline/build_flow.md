# Bare-Metal MLIR-AIE Build Pipeline (WSL + Windows + XRT)

This documents the end-to-end build flow for running custom AIE kernels on AMD Ryzen AI NPUs from a Windows development machine.

## Overview

```
┌──────────────────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│  WSL2 (Ubuntu)           │     │  Windows (MSVC)          │     │  NPU Hardware    │
│                          │     │                          │     │                  │
│  1. Peano/clang++        │     │  3. cl.exe compiles      │     │  4. XRT submits  │
│     compiles AIE C++     │────▶│     host C++ with        │────▶│     xclbin +     │
│     → kernel .o files    │     │     XRT SDK → .exe       │     │     instructions │
│                          │     │                          │     │     to NPU       │
│  2. IRON design.py       │     │                          │     │                  │
│     generates MLIR       │     │                          │     │                  │
│     → aiecc.py compiles  │     │                          │     │                  │
│     → xclbin + insts.bin │     │                          │     │                  │
└──────────────────────────┘     └──────────────────────────┘     └─────────────────┘
```

## Prerequisites

### WSL2 Environment
- **Distro**: Ubuntu (standard, not Ubuntu-24.04 variant)
- **MLIR-AIE**: Installed via `ironenv` venv at `~/mlir-aie/ironenv/`
- **IRON**: Cloned at `~/IRON/`
- **Peano compiler**: `ironenv/lib/python3.12/site-packages/llvm-aie/bin/clang++`
- **MLIR-AIE package**: `ironenv/lib/python3.12/site-packages/mlir_aie/`

### Windows Environment
- **Visual Studio Build Tools 2022**: For `cl.exe` (C++ compiler)
- **XRT SDK**: Headers and `xrt_coreutil.lib` for linking
- **NPU driver**: AMD NPU driver for XDNA2 hardware

## Step 1: Compile AIE Kernels (WSL)

AIE compute kernels are written in C++ using the AIE API:

```bash
source ~/mlir-aie/ironenv/bin/activate

PEANO_DIR=$(python -c "import aie.utils.config; print(aie.utils.config.peano_install_dir())")
MLIR_AIE_DIR=$(python -c "import aie.utils.config; print(aie.utils.config.root_path())")

# Compile a kernel
$PEANO_DIR/bin/clang++ \
    -O2 --target=aie2p-none-unknown-elf -std=c++20 \
    -I $MLIR_AIE_DIR/include/ \
    -DDIM_M=32 -DDIM_K=64 -DDIM_N=64 \
    -c kernel.cc -o kernel.o
```

**Critical flags**:
- Target: `aie2p-none-unknown-elf` (NOT `aie2p-none-elf` — causes `__config_site` error)
- Must use `-std=c++20` and `-I $MLIR_AIE_DIR/include/`
- BFP16 emulation: add `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` if desired

## Step 2: Generate MLIR and Compile to xclbin (WSL)

IRON's `design.py` scripts generate MLIR describing the tile layout and data movement:

```bash
# Generate MLIR from IRON operator
python $IRON_DIR/iron/operators/gemm/design.py \
    --dev npu2 \
    -M 128 -K 768 -N 768 \
    -m 32 -k 64 -n 64 \
    --n-aie-cols 4 \
    --emulate-bf16-mmul-with-bfp16 \
    --prio-accuracy \
    --archive kernels.a \
    -o gemm.mlir

# Compile MLIR to xclbin + instruction binary
python $MLIR_AIE_DIR/bin/aiecc.py \
    --no-xchesscc --no-xbridge \
    --peano $PEANO_DIR \
    --dynamic-objFifos \
    --aie-generate-xclbin \
    --xclbin-name=gemm.xclbin \
    --xclbin-kernel-name=MLIR_AIE \
    --aie-generate-npu-insts \
    --npu-insts-name=gemm.bin \
    gemm.mlir
```

**Critical flags**:
- Use `--no-xchesscc --no-xbridge --peano=<path>` (NOT `--no-compile` which skips ELF generation)
- `--dynamic-objFifos` enables IRON's ObjectFIFO patterns
- Kernel name (`MLIR_AIE`) must match what the host code uses

**Output**: `gemm.xclbin` (device binary) + `gemm.bin` (NPU instruction sequence)

## Step 3: Build Host Application (Windows)

The host application uses XRT C++ API to load xclbin and submit work:

```bat
cl.exe /EHsc /std:c++20 /O2 /arch:AVX2 ^
    /I"host\include" ^
    app.cpp test_utils.cpp ^
    /Fe:"app.exe" ^
    /link /LIBPATH:"xrt-lib" xrt_coreutil.lib
```

Host code pattern:
```cpp
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

// Open device, load xclbin
xrt::device device(0);
auto xclbin = xrt::xclbin("gemm.xclbin");
device.register_xclbin(xclbin);
auto ctx = xrt::hw_context(device, xclbin.get_uuid());
auto kernel = xrt::kernel(ctx, "MLIR_AIE");

// Load instructions
auto bo_instr = xrt::bo(device, instr_size, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
// ... copy instruction data, sync to device

// Allocate data buffers
auto bo_a = xrt::bo(device, size_a, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
auto bo_b = xrt::bo(device, size_b, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
auto bo_c = xrt::bo(device, size_c, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

// Upload data
memcpy(bo_a.map<void*>(), data_a, size_a);
bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

// Run
auto run = kernel(3, bo_instr, instr.size(), bo_a, bo_b, bo_c);
run.wait();

// Read results
bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
```

## Step 4: Run on NPU

```bash
# From Windows (Git Bash or cmd)
./app.exe --kernel MLIR_AIE
```

**Important**: Always pass `--kernel MLIR_AIE` to match the kernel name in the xclbin.

## Gotchas and Lessons Learned

### WSL ↔ Windows Path Issues
- Set `MSYS_NO_PATHCONV=1` before `wsl` commands from Git Bash
- WSL paths: `/mnt/c/Users/...` maps to `C:\Users\...`

### AIE2P Hardware Specifics
- `aie::mmul<4,8,8>` is the correct BF16 shape on AIE2P
- Scalar BF16 loops are broken — always use `aie::` vector ops or mmul
- DMA channel limit: 2 input + 2 output per tile
- Shim DMA limit: max 2 fills per host arg (group_id) — 3 fills causes zeros

### Buffer Allocation
- Minimum BO size: 4096 bytes (even for smaller data)
- Use `XRT_BO_FLAGS_HOST_ONLY` for data buffers
- Use `XCL_BO_FLAGS_CACHEABLE` for instruction buffers
- Pre-allocate and reuse BOs — allocation has overhead

### IRON MHA Buffer Layout
- Q, K, V, O all use layout `[Heads, Seq, HeadDim]`
- 4 XRT args (group_id 3, 4, 5, 6)
- Requires data repack from standard `[Seq, Dim]` interleaved format

### Performance Tips
- XRT submission overhead: ~0.4ms fixed. Don't over-optimize for fewer submissions.
- Elementwise ops (GELU, ReLU, Add) are faster on CPU due to DMA overhead.
- Weight BOs can be uploaded once and reused across all inference calls.
- 4-col (16 tiles) vs 8-col (32 tiles): minimal difference when memory-bound.
