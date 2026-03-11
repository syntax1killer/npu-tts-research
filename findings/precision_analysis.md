# BF16/BFP16 Precision Analysis: Iterative ALBERT on AMD AIE2P NPU

**Target hardware**: AMD AIE2P (Phoenix/Hawk Point NPU)
**Model**: Kokoro TTS -- ALBERT encoder (1 shared transformer layer, 12 iterative passes)
**Dimensions**: hidden=768, heads=12, FFN intermediate=2048
**Date**: 2026-03

---

## 1. Problem Statement

Kokoro TTS uses an ALBERT encoder: a single shared transformer layer applied iteratively 12 times, with the output of each pass fed back as input to the next. This weight-sharing design reduces model size but creates a precision feedback loop that is catastrophic on hardware with limited floating-point formats.

The AMD AIE2P `aie::mmul` intrinsic accepts only BF16 inputs. There is no FP32 input path. The `prio_accuracy` mode enables FP32 accumulation within the multiply-accumulate pipeline, but the operand inputs are still truncated to BF16 (8-bit exponent, 7-bit mantissa, ~3.9 decimal digits). Every intermediate result stored between passes is likewise truncated from FP32 to BF16, losing approximately 16 mantissa bits each time.

In a standard 12-layer transformer with independent weights, each layer's BF16 truncation error is uncorrelated and does not systematically compound. In ALBERT's iterative architecture, the error feeds back through the same weights, creating a correlated amplification loop across passes.

---

## 2. Architecture Under Test

Each ALBERT pass executes the following POST-norm sequence:

```
Input (768-dim)
  -> GEMM_Q (768x768) + GEMM_K (768x768) + GEMM_V (768x768)
  -> Multi-Head Attention (12 heads, head_dim=64)
  -> Attention Output Dense GEMM (768x768)
  -> Residual Add + LayerNorm
  -> FFN Up GEMM (768x2048) + GELU
  -> FFN Down GEMM (2048x768)
  -> Residual Add + LayerNorm
  -> Output (768-dim) -- fed back as next pass input
```

Each pass contains 6 GEMM operations and 1 multi-head attention (softmax + batched matmul). Across 12 passes, this yields 72 GEMM truncation events and 12 MHA truncation events, all feeding back through the same weight matrices.

**Weight inventory**: 6 weight matrices (4 at 768x768, 1 at 768x2048, 1 at 2048x768), 6 bias vectors, 4 LayerNorm parameter vectors (2 per LN). Total: 10.5 MB in 16 binary files.

---

## 3. Methodology

### 3.1 Input Extraction

Test inputs were extracted from a real ONNX Runtime FP32 inference session, not generated randomly. This distinction is critical: random inputs do not exercise the attention mechanism realistically (attention patterns over random data are near-uniform, masking precision issues in softmax and value aggregation). Real inputs contain structured, sparse attention patterns that are sensitive to small numerical perturbations.

Extraction tool: `kokoro/extract_albert_io.py` -- hooks the ALBERT subgraph in the full Kokoro ONNX model and dumps the 768-dim input tensor as FP32 binary.

### 3.2 NPU Execution

The ALBERT benchmark loads real extracted weights (BF16-converted from FP32 ONNX weights via `kokoro/extract_weights.py`) and executes 12 iterative passes on the NPU. Each pass dumps its output as a BF16 binary for per-pass comparison.

Build flow: WSL2 (MLIR-AIE -> xclbin + insts.bin) -> Windows MSVC host executable -> XRT runtime on NPU hardware.

### 3.3 Reference Comparison

Per-pass outputs are compared against FP32 ONNX Runtime reference outputs using Pearson correlation. This captures both magnitude drift and directional divergence across the 768-dim hidden state.

Tool: `kokoro/compare_precision.py`

### 3.4 End-to-End Audio Evaluation

The final ALBERT output (pass 11) is injected back into the full Kokoro ONNX model, which runs the remaining stages (duration predictor, decoder, vocoder) in FP32 on CPU. The resulting audio waveform is compared to the FP32-only reference audio via:

- Waveform correlation (Pearson)
- Signal-to-Noise Ratio (SNR in dB)
- Subjective listening comparison

Tool: `kokoro/audio_compare.py`

---

## 4. Results

### 4.1 Per-Pass Correlation Degradation

With BFP16 GEMM + BFP16 IRON MHA (the fastest NPU-only configuration):

| Pass | Correlation vs FP32 |
|------|---------------------|
| 0    | 0.9993              |
| 1    | 0.9977              |
| 2    | 0.9949              |
| 3    | 0.9910              |
| 4    | 0.9860              |
| 5    | 0.9797              |
| 6    | 0.9720              |
| 7    | 0.9541              |
| 8    | 0.9588              |
| 9    | 0.9631              |
| 10   | 0.9684              |
| 11   | 0.9736              |

Correlation drops steadily through pass 7, then partially recovers. The recovery is likely due to LayerNorm re-centering the distribution after the error has saturated certain dimensions. Final correlation of 0.698 reported below is measured differently (full tensor comparison including all sequence positions), not just the aggregated per-pass metric.

### 4.2 Configuration Comparison

| Configuration | Final Corr | Latency (ms) | Audio Quality |
|---|---|---|---|
| BFP16 GEMM + BFP16 IRON MHA | 0.698 | 244 | Destroyed |
| BFP16 GEMM + Native BF16 MHA (no BFP16 block quant) | 0.343 | 275 | Destroyed |
| BFP16 GEMM + CPU FP32 MHA (naive scalar) | 0.974 | 457 | Duration mismatch |
| BFP16 GEMM + CPU FP32 MHA (AVX2 vectorized) | 0.973 | 229 | Duration mismatch |
| Native BF16 GEMM + CPU FP32 MHA (AVX2) | 0.974 | 266 | Duration mismatch |
| BF16-simulated pure CPU (storage only, FP32 compute) | 0.9999 | -- | Fine |
| FP32 ONNX Runtime (reference) | 1.000 | ~300 | Reference |

**Latency notes**: The NPU-only configuration (244ms for 12 ALBERT passes) is 1.23x faster than the CPU FP32 reference (~300ms). The hybrid NPU GEMM + CPU AVX2 MHA configuration (229ms) is even faster due to efficient AVX2 attention.

### 4.3 Error Source Isolation

The configuration matrix isolates two independent error sources:

**MHA is the dominant error source.** Replacing only MHA with CPU FP32 (keeping BFP16 GEMM on NPU) raises correlation from 0.698 to 0.974. This is a larger improvement than any GEMM-side change.

**BFP16 block quantization is worse than element-wise BF16 for MHA, but paradoxically better in aggregate.** Native BF16 MHA (element-wise truncation, no block quantization) yields 0.343 correlation -- far worse than BFP16 MHA's 0.698. This is counterintuitive: BFP16 shares an exponent across a block of values, which should introduce more error than element-wise BF16. The likely explanation is that BFP16's shared exponent preserves relative magnitudes within attention score blocks, while element-wise BF16 truncation independently distorts each score, destroying the softmax distribution more severely.

**BFP16 vs native BF16 for GEMM: no measurable difference.** Both yield 0.974 correlation when MHA runs on CPU. GEMM error is dominated by the BF16 storage format between passes, not by the compute-time quantization scheme.

**BF16 storage is the residual error source.** The BF16-simulated CPU test (which converts to BF16 for storage between passes but computes everything in FP32) achieves 0.9999 correlation. The gap from 0.974 to 0.9999 is entirely due to BF16 truncation of GEMM inputs and outputs, accumulated over 72 GEMM operations across 12 passes.

---

## 5. Audio Impact Analysis

### 5.1 Duration Predictor Sensitivity

Kokoro's duration predictor is a stack of 6 LSTMs that consume the ALBERT encoder output and predict per-phoneme durations (in mel frames). This component is extremely sensitive to the ALBERT output distribution.

Even at 0.974 correlation (the best NPU-achievable configuration), the predicted phoneme durations diverge sufficiently to produce audio of a different total length: ~120,000 samples vs ~106,000 samples in the reference. This duration mismatch renders waveform-level comparison metrics (correlation, SNR) meaningless, as the waveforms are no longer temporally aligned.

### 5.2 Audio Metrics

For the best NPU configuration (BFP16 GEMM + CPU FP32 MHA, 0.974 correlation):

- Audio waveform correlation: **0.01** (uncorrelated due to length mismatch)
- Signal-to-Noise Ratio: **-10.5 dB**
- Subjective quality: Intelligible but noticeably different prosody; unusable as a drop-in replacement

For the full-NPU configuration (0.698 correlation):

- Audio: completely destroyed; unintelligible

### 5.3 The 0.974 Threshold Is Insufficient

A 0.974 Pearson correlation on a 768-dim hidden state sounds high, but it represents a per-dimension average error that, when fed through a sensitive downstream LSTM, produces meaningfully different predictions. The duration predictor acts as an error amplifier: small distributional shifts in the ALBERT output are converted into discrete frame-count differences that cascade into large waveform divergence.

---

## 6. Root Cause Analysis

### 6.1 Hardware Constraint

The AIE2P `aie::mmul<4,8,8>` intrinsic for BF16 is the only GEMM primitive available on this hardware. There is no FP32 input path. The `prio_accuracy` accumulation mode mitigates within-tile error but cannot prevent input truncation. Every matrix multiplication on the NPU inherently operates on 7-bit mantissa inputs.

### 6.2 Iterative Amplification

The fundamental issue is the interaction between BF16 precision and iterative weight-sharing:

```
Per-GEMM BF16 truncation error: ~2^-8 relative
GEMMs per pass: 6
Passes: 12
Total truncation events: 72
Compounding factor: errors are correlated (same weights amplify same error modes)
```

In a standard 12-layer transformer, each layer has independent weights that project the representation into different subspaces. BF16 errors in layer N are not preferentially amplified by layer N+1's weights. In ALBERT, the same weight matrices see their own truncation artifacts 12 times, creating resonant error modes in the representation.

### 6.3 Why BF16 Storage Alone Is Fine

The BF16-simulated CPU test (0.9999 correlation) proves that BF16 as a storage format between passes is acceptable. The 7-bit mantissa provides sufficient precision for intermediate representations when all compute is done in FP32. The problem is specifically BF16 truncation at GEMM inputs, where the truncated values are multiplied together (squaring the relative error) and accumulated across 768-element dot products.

---

## 7. Considered Mitigations

| Mitigation | Feasibility | Expected Impact |
|---|---|---|
| FP32 GEMM on NPU | Impossible -- hardware has no FP32 mmul | Would solve the problem |
| Mixed-precision: NPU GEMM + CPU MHA | Implemented (229ms) | 0.974 corr -- insufficient |
| Stochastic rounding on BF16 conversion | Not available in AIE2P ISA | Would reduce systematic bias |
| Kahan summation in residual connections | Possible in CPU-side Add | Marginal improvement expected |
| Error-compensated weight sharing (store correction term) | Would require model retraining | Could work but out of scope |
| Run ALBERT on CPU, NPU for decoder/vocoder | Viable | ALBERT is only 300ms on CPU |
| Fine-tune model with BF16-aware training | Requires training infrastructure | Best long-term solution |
| Replace ALBERT with standard 12-layer transformer | Requires retraining, larger model | Eliminates iterative amplification |

---

## 8. Conclusions

### 8.1 BF16 Is Incompatible with Iterative Weight-Sharing at This Depth

Twelve iterative passes through BF16-truncated GEMMs produce output that is too imprecise for Kokoro's downstream duration predictor. The error is not in any single component but in the cumulative effect of 72 truncated matrix multiplications feeding back through shared weights.

### 8.2 The Bottleneck Is GEMM Input Truncation, Not Accumulation

FP32 accumulation (`prio_accuracy`) is necessary but insufficient. The dominant error path is: FP32 result -> BF16 storage -> BF16 GEMM input -> truncation at multiply. Improving accumulation precision (already FP32) provides no further benefit.

### 8.3 MHA Precision Is Critical but Not Sufficient

Moving MHA to FP32 raises correlation from 0.698 to 0.974, which is a large improvement but still below the threshold needed for acceptable audio. The remaining 0.974-to-1.000 gap from GEMM BF16 truncation is enough to break the duration predictor.

### 8.4 Non-Iterative Components Are Viable on NPU

The Conv1d decoder, HiFi-GAN vocoder, and other non-iterative pipeline stages do not suffer from error compounding. BF16/BFP16 precision is adequate for single-pass or feed-forward architectures where errors do not feed back through shared weights. These components remain strong candidates for NPU acceleration.

### 8.5 Recommendations for NPU Deployment

1. **Run ALBERT on CPU** (300ms FP32 baseline). The NPU's 1.23x speedup is negated by unusable output quality.
2. **Target NPU acceleration at the decoder and vocoder**, where single-pass convolutions tolerate BF16 precision.
3. **For future iterative architectures on BF16 hardware**: validate precision with real model inputs after the full iteration count. Per-pass error metrics are misleading -- a 0.9993 single-pass correlation compounding over 12 passes does not yield 0.9993^12; correlated errors through shared weights amplify nonlinearly.
4. **Test with real inputs, not random data.** Random inputs produce near-uniform attention distributions that mask precision issues in softmax. Real inputs with peaked, sparse attention patterns are far more sensitive to BF16 truncation.

---

## Appendix A: Tool Inventory

| Tool | Purpose |
|---|---|
| `kokoro/extract_weights.py` | Extract ALBERT weights from ONNX model to 16 binary files |
| `kokoro/extract_albert_io.py` | Capture real ALBERT input/output from ONNX Runtime session |
| `kokoro/compare_precision.py` | Per-pass correlation comparison (NPU vs FP32 reference) |
| `kokoro/audio_compare.py` | Waveform correlation, SNR, and duration comparison |
| `kokoro/weights/manifest.json` | Weight file manifest (shapes, dtypes, file paths) |

## Appendix B: Hardware Configuration

- **NPU**: AMD AIE2P (integrated in Phoenix/Hawk Point APU)
- **Compute primitive**: `aie::mmul<4,8,8>` with BF16 inputs, FP32 accumulation
- **Block float**: BFP16 (shared 8-bit exponent per block of values)
- **Runtime**: Xilinx XRT (xclbin-based kernel dispatch)
- **Build**: MLIR-AIE (IRON flow) on WSL2, MSVC host on Windows
- **CPU reference**: FP32 ONNX Runtime, AVX2 attention kernel for hybrid tests
