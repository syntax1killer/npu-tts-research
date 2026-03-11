# What Didn't Work: NPU TTS Acceleration Failed Approaches

This document catalogs every significant failed approach during the AMD XDNA 2 NPU TTS acceleration research project (Kokoro TTS on OneXPlayer G1, Ryzen AI 9 HX 370). Each entry explains what was tried, the measured result, and the root cause of failure.

The goal is to save other researchers from repeating these dead ends.

---

## 1. VitisAI ONNX Runtime (Automatic NPU Offload)

**What we tried:** Use AMD's VitisAI execution provider with the Ryzen AI SDK 1.7.0 BF16/VAIML path to automatically partition the Kokoro ONNX model onto the NPU. The idea was zero custom kernel work -- just point the ONNX Runtime session at the NPU and let the compiler figure out placement.

**What happened:** The VAIML compiler split the model into 36 subgraphs. Of 2082 total ops, only 1080 were NPU-supported. The remaining ops (Gather, Shape, Unsqueeze, Cast, LSTM, etc.) fell back to CPU, creating 36 CPU-NPU transition boundaries. Final result: **RTF 0.314x -- identical to CPU-only baseline (0.31x)**. No speedup whatsoever.

**Why it failed:** Kokoro's hybrid architecture (ALBERT transformer + LSTM duration predictor + Conv1d decoder + iSTFT vocoder) uses too many op types the NPU doesn't support. Each unsupported op creates a subgraph boundary requiring a CPU-NPU data transfer. With 36 boundaries, the DMA transfer overhead completely negated the compute savings on the NPU tiles. The VitisAI partitioner has no way around this -- it's a fundamental mismatch between the model's op diversity and the NPU's supported op set.

**Lesson:** Automatic NPU offload only works for homogeneous models (e.g., pure Conv+InstanceNorm+ReLU stacks). Hybrid architectures mixing transformers, RNNs, and convolutions will always fragment badly on current VitisAI.

---

## 2. INT8 Quantization via ONNX Runtime

**What we tried:** Quantize Kokoro to INT8 using ORT's quantization tools, both dynamic and static, for potential NPU deployment via the X2 (INT8) path.

**Results:**
- Dynamic quantization: **SNR -10.1 dB** (audio destroyed)
- Static quantization with calibration data: **SNR -4.0 dB** (still terrible)

**Why it failed:** The calibration data was random phoneme token sequences. These random inputs caused the FP32 model to overflow in certain paths, producing garbage activation statistics. The calibration was fundamentally invalid. ORT's quantization tools don't warn you when calibration produces nonsensical scale factors.

**Additional finding:** INT8 turned out to be unnecessary anyway. IRON BFP16 GEMM performance was 20-65x better than initial projections (0.32ms per GEMM vs. projected 6.6ms). The BFP16 path achieved 1.23x faster than CPU without INT8.

**Lesson:** Never calibrate quantization with random inputs -- use real representative data from the actual inference pipeline. Also, measure your baseline performance before committing to a quantization effort; you may not need it.

---

## 3. Removing BFP16 Flag from IRON MHA

**What we tried:** Build the IRON MHA (Multi-Head Attention) kernel with `--emulate-bf16-mmul-with-bfp16 False` to use native BF16 `(4,8,8)` matrix multiply intrinsics instead of BFP16 `(8,8,8)` emulated multiply. The hypothesis was that native BF16 compute would be more precise than BFP16 block quantization.

**What happened:** Correlation dropped from **0.698 to 0.343** -- dramatically worse, not better.

**Why it failed:** The `--emulate-bf16-mmul-with-bfp16` flag doesn't just swap the mmul intrinsic shape. It controls the entire MLIR code generation path in `design.py` -- memory layouts, tiling strategies, and DMA routing patterns all change. The BFP16 mode's data flow happened to produce more numerically stable intermediate results for the attention computation. Removing the flag didn't just change precision of individual multiplies; it changed how data moved through the tile array.

**Lesson:** Compiler flags in MLIR-AIE/IRON can have non-obvious emergent effects. "Removing quantization" does not always improve precision when the flag controls code generation beyond the arithmetic itself. Always benchmark both configurations rather than assuming.

---

## 4. Removing BFP16 Flag from IRON GEMM

**What we tried:** Build IRON GEMM without BFP16 emulation while keeping `prio_accuracy` (FP32 accumulation) enabled. Goal was to eliminate the BFP16 block quantization error that we suspected was degrading precision across 12 iterative ALBERT passes.

**What happened:** Correlation was **0.974 -- identical to the BFP16 GEMM result**. No improvement.

**Why it failed:** The error source was not BFP16 block quantization during compute. It was BF16 storage format truncation between passes. Each activation stored as BF16 between ALBERT passes loses approximately 8 mantissa bits compared to FP32. Over 12 iterative passes, this truncation compounds: pass 0 correlation was 0.96, degrading roughly 2% per pass, reaching 0.70 after all 12 passes. Since both BFP16 and non-BFP16 modes store intermediate results in BF16, they produce the same error.

**Confirmation:** BF16 storage alone tested at 0.9999 correlation per pass -- but the cumulative effect across 12 passes destroyed the signal. Final audio output: correlation 0.01, SNR -10.5 dB.

**Lesson:** When two compute modes (BFP16 and non-BFP16) produce the same result, the error source is upstream of the compute -- likely in the data format, not the arithmetic. Identify the error source before optimizing the wrong thing.

---

## 5. NPU GELU (Standalone Elementwise Kernel)

**What we tried:** Run the GELU activation function on NPU instead of CPU, to keep data on-device and avoid a CPU round-trip in the ALBERT pipeline. The activation operates on 128x2048 = 262,144 BF16 values (512 KB).

**Results:**
- NPU GELU: **1.99 ms/pass**
- CPU GELU: **1.46 ms/pass**
- NPU was **36% slower** than CPU.

**Why it failed:** Elementwise operations like GELU have arithmetic intensity of approximately 1 op/byte -- far too low for NPU offload to be worthwhile. The DMA round-trip cost (host memory to NPU tiles and back) for 512 KB of data exceeds the compute time. The NPU's advantage is massive parallelism on high-arithmetic-intensity operations; with only one operation per data element, the compute is trivial and the transfer dominates.

**Lesson:** Only offload operations with high arithmetic intensity to the NPU. GEMM has ~100+ ops/byte and sees massive speedups (248x in our tests). Elementwise ops (GELU, ReLU, Add, LayerNorm) are faster on CPU unless they can be fused into a larger kernel pipeline that amortizes the DMA cost.

---

## 6. Fused QKV GEMM (3-in-1 Matrix Multiply)

**What we tried:** Replace 3 separate 128x768x768 GEMMs (for Q, K, V projection in attention) with a single fused 128x768x2304 GEMM. The goal was to eliminate 2 kernel dispatch overheads and reduce the 3 * ~4.2ms = 12.6ms down to a single dispatch.

**What happened:** Total ALBERT time went from **246ms to 243ms**. Saved only **3ms** across 12 passes, not the projected 24ms.

**Why it failed:** IRON GEMM dispatch overhead was only ~1.4ms per call -- much lower than initially assumed. Eliminating 2 dispatches saved ~2.8ms, which is exactly what we measured. Additionally, the larger fused GEMM had slightly worse per-element performance due to different tiling factors at the 2304-wide output dimension. The net saving was real but minimal.

**Lesson:** Measure dispatch overhead before optimizing for fewer dispatches. If the runtime is already spending only ~7% on dispatch overhead, fusion for dispatch reduction has diminishing returns. Profile first, optimize second.

---

## 7. Random Input Precision Testing

**What we tried:** Validate NPU ALBERT precision using random BF16 input vectors, comparing NPU output against CPU FP32 reference output pass-by-pass.

**What happened:** Correlation numbers were inconsistent and did not predict real-world audio quality. The tests suggested precision was "acceptable" when it was not.

**Why it failed:** Random inputs produce unrealistic attention patterns. When Q, K, V projections are computed from random embeddings, the resulting QK^T scores are roughly uniform. Softmax on near-uniform scores gives near-uniform attention weights, which masks the precision issues that real structured embeddings would expose. Real text embeddings have high-variance, structured attention patterns where small numerical errors in softmax get amplified.

**What we did instead:** Built `extract_albert_io.py` to extract the actual ALBERT input tensor from ONNX Runtime's reference execution of the full Kokoro model. Testing with this real input gave definitive results: 0.96 correlation at pass 0 degrading to 0.70 at pass 12, with final audio at correlation 0.01 and SNR -10.5 dB.

**Lesson:** Always validate precision with real model inputs extracted from the reference implementation. Synthetic/random inputs can give misleadingly optimistic results for attention-based architectures.

---

## 8. L2 Weight Preloading (Analyzed, Not Attempted)

**What we considered:** Cache ALBERT's shared weights in the NPU's L2 memory (Mem tiles, 4 MB total) to avoid re-streaming from DDR on each of the 12 passes.

**Why we didn't attempt it:**
- ALBERT weights are **14 MB in BF16**. L2 capacity is **4 MB**. Maximum cacheable: ~28% of weights.
- Even caching 28% would save only ~29% of DDR bandwidth for the ALBERT portion.
- Implementation would require custom ObjectFIFO patterns that fight against IRON's design philosophy, with high engineering complexity.
- Alternative approaches (BFP16 bandwidth reduction, which is built into IRON GEMM) achieved similar bandwidth savings with zero custom work.

**Lesson:** Do the capacity math before attempting caching strategies. If the working set exceeds cache by 3.5x, the achievable benefit is modest and the complexity is high.

---

## Summary Table

| # | Approach | Expected Outcome | Actual Result | Root Cause |
|---|----------|-----------------|---------------|------------|
| 1 | VitisAI auto-offload | Fast NPU inference | RTF 0.314 (no speedup) | 36 subgraphs from unsupported ops |
| 2 | INT8 quantization | Smaller/faster model | SNR -4.0 dB (unusable) | Bad calibration data + unnecessary |
| 3 | Remove BFP16 from MHA | Better precision | Correlation 0.343 (worse) | Flag controls codegen, not just precision |
| 4 | Remove BFP16 from GEMM | Better precision | Correlation 0.974 (no change) | Error is BF16 storage, not BFP16 compute |
| 5 | NPU GELU | Faster activation | 1.99ms vs 1.46ms CPU (slower) | DMA overhead > compute for elementwise |
| 6 | Fused QKV GEMM | Save ~24ms | Saved 3ms | Dispatch overhead already only 1.4ms |
| 7 | Random input testing | Validate precision | Misleading results | Random inputs mask attention precision issues |
| 8 | L2 weight caching | Reduce DDR traffic | Not attempted | 14MB weights vs 4MB L2 = poor ROI |

---

## What DID Work

For completeness, the approaches that succeeded are documented elsewhere in the project:

- **Bare-metal MLIR-AIE/IRON GEMM** achieved 248x speedup over naive NPU GEMM (0.32ms per matmul) with BFP16 + FP32 accumulation.
- **Full ALBERT on NPU** completed 12 passes in 243.82ms -- 1.23x faster than the 300ms CPU baseline.
- **IRON MHA** processed 12 attention heads in 1.66ms standalone (3.94ms/pass integrated with data repacking).

The precision problem (BF16 compound error across 12 passes) remains the primary open challenge.
