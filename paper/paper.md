# BF16 Precision Limits in Iterative Neural Architectures on AMD XDNA2 NPU

**Authors:** syntax1killer (Independent Researcher)

**Abstract.** Neural Processing Units (NPUs) in consumer processors promise efficient on-device AI inference through low-precision matrix units, but their numerical limitations remain poorly characterized for iterative architectures. We present the first bare-metal characterization of BF16 precision through an iterative weight-sharing architecture — the ALBERT encoder in Kokoro TTS — on AMD XDNA2 (AIE2P) hardware. Using the IRON close-to-metal programming framework, we bypass the vendor auto-partitioning toolchain (which fragments Kokoro into 36 CPU-NPU subgraphs with zero speedup) and achieve 1.31x speedup over CPU (229ms vs 300ms for 12 ALBERT passes). However, we find that BF16 input truncation in the hardware matrix unit (`aie::mmul`) compounds through 12 iterative passes via correlated error amplification, degrading Pearson correlation from 0.9993 (single pass) to 0.698 (full NPU) or 0.974 (best hybrid). Even at 0.974 correlation, the downstream duration predictor produces different phoneme timings, yielding audio with 11.3% duration mismatch and -10.5 dB SNR — functionally unusable. We isolate three error sources (BF16 storage, GEMM input truncation, MHA block quantization) and demonstrate that MHA accounts for 80% of precision loss while GEMM truncation creates an irreducible 0.974 ceiling. INT8 quantization via AMD Quark produces even worse results (-15.6 dB SNR). Our findings establish that BF16-only matrix units are fundamentally incompatible with iterative weight-sharing at depth 12+, with implications for deploying diffusion models, Universal Transformers, and other iterative architectures on current-generation NPU hardware.

---

## 1. Introduction

Neural Processing Units are proliferating across consumer hardware. AMD's XDNA2 NPU (branded Ryzen AI) ships in every Ryzen AI 300-series processor, offering 50 TOPS INT8 and 14.7 TFLOPS BF16 through a 4x8 array of AI Engine tiles. Intel's Meteor Lake and Qualcomm's Hexagon include similar dedicated accelerators. The promise is clear: efficient, low-power inference for always-on AI workloads without taxing the CPU or GPU.

These NPUs achieve high throughput through reduced-precision arithmetic. AMD's AIE2P tiles provide a `aie::mmul` intrinsic that accepts only BF16 (bfloat16) inputs — 8-bit exponent, 7-bit mantissa, approximately 3.9 decimal digits of precision. FP32 accumulation is available during the multiply-accumulate operation, but operand inputs are irrecoverably truncated. For single-pass architectures (standard transformers, CNNs), this truncation is well-tolerated: each layer's error is small and largely uncorrelated with subsequent layers' errors.

However, a growing class of neural architectures reuse the same weights iteratively. ALBERT [1] applies a single shared transformer layer N times in sequence, feeding each pass's output back as the next pass's input. Diffusion models [2] apply a shared UNet through 16-50 denoising steps. Universal Transformers [3] iterate until convergence. In these architectures, BF16 truncation errors do not stay independent — they feed back through the same weight matrices, creating correlated error amplification.

We investigate this interaction empirically using Kokoro TTS [4], an 82M-parameter text-to-speech model that uses an ALBERT encoder (1 shared transformer layer, 12 passes, hidden dimension 768) followed by LSTM duration predictors, a convolutional decoder, and a HiFi-GAN vocoder. We program the AMD XDNA2 NPU at the bare-metal level using MLIR-AIE and IRON [5], bypassing the vendor VitisAI auto-partitioning toolchain which produces 36 CPU-NPU subgraphs with no measurable speedup.

**Contributions.** (1) We demonstrate that BF16 input truncation compounds through iterative weight-sharing, producing correlation degradation from 0.9993 to 0.698 across 12 passes. (2) We isolate three independent error sources and quantify their contributions. (3) We show that even the best hybrid configuration (0.974 correlation) fails a sensitive downstream consumer (duration prediction), despite appearing numerically adequate. (4) We provide the first bare-metal XDNA2 performance characterization for a complete transformer workload, achieving 229ms through 6 optimization steps from an initial 1022ms. (5) We establish practical guidelines for deploying iterative architectures on BF16-only accelerators.

## 2. Background

### 2.1 AMD XDNA2 Architecture

The AMD XDNA2 NPU (codenamed Strix) integrates an array of AI Engine 2P (AIE2P) tiles. Each tile contains a vector processor with a hardware matrix multiply unit accessed through the `aie::mmul<M,K,N>` intrinsic. For BF16, the supported shape is `aie::mmul<4,8,8>`, processing a 4x8 BF16 matrix multiplied by an 8x8 BF16 matrix per cycle. The `prio_accuracy` mode enables FP32 accumulation within the multiply-accumulate pipeline, but the input operands are still truncated to BF16 before multiplication.

The NPU communicates with the host via DDR through a shim DMA layer. Kernel dispatch uses the Xilinx XRT runtime with `.xclbin` bitstream files compiled from MLIR-AIE. Each XRT submission incurs approximately 0.4ms of fixed overhead regardless of compute workload.

Two programming paths exist: (1) VitisAI, a high-level auto-partitioning framework that compiles ONNX models by fragmenting unsupported operations to CPU, and (2) IRON/MLIR-AIE, a close-to-metal framework that programs individual tiles and data movement explicitly. We use path (2) because path (1) fragments Kokoro into 36 subgraphs (Section 3.1).

### 2.2 BF16 Numerical Properties

BF16 (bfloat16) uses an 8-bit exponent and 7-bit mantissa, providing the same dynamic range as FP32 but with approximately 3.9 decimal digits of precision versus FP32's 7.2. Truncation from FP32 to BF16 introduces a relative error of up to 2^-8 per value. When two BF16 values are multiplied, the relative error in the product is approximately 2^-7 (errors add in multiplication). Across a dot product of dimension K, the accumulated error grows as O(sqrt(K)) with random errors or O(K) with systematic bias.

Block floating point (BFP16) shares a single 8-bit exponent across a block of values, storing only the 7-bit mantissa per element. This reduces storage but constrains the dynamic range within each block. Counter-intuitively, we find BFP16 can preserve relative magnitudes within blocks better than element-wise BF16 in certain operations (Section 4.2).

### 2.3 ALBERT and Iterative Weight-Sharing

ALBERT (A Lite BERT) [1] reduces transformer parameters by sharing a single transformer layer across all N passes. The output of pass i becomes the input of pass i+1, with the same weight matrices (Q, K, V, FFN) applied each time. This creates a parameter-efficient but computationally iterative architecture: N passes through the same weights, with the representation evolving through the shared subspace at each step.

In a standard N-layer transformer, each layer's weights project the representation into a different subspace. BF16 truncation errors at layer i are not preferentially amplified by layer i+1's independent weights. In ALBERT, the same weights see their own truncation artifacts N times, creating resonant error modes — truncation errors that align with the weight matrices' principal components are amplified at each pass.

### 2.4 Kokoro TTS

Kokoro [4] is an 82M-parameter text-to-speech system with the following pipeline:

1. **ALBERT Encoder** (12 passes, dim=768): Converts phoneme tokens to hidden representations
2. **Duration Predictor** (6 LSTMs): Predicts per-phoneme duration in mel frames
3. **Decoder** (Conv1d + AdaIN): Generates mel spectrograms
4. **Vocoder** (HiFi-GAN with LeakyReLU): Converts mel spectrograms to audio waveforms

The duration predictor is the critical interface between the encoder and the rest of the pipeline. It consumes the 768-dimensional ALBERT output and produces discrete frame counts per phoneme. Small distributional shifts in the ALBERT output can produce different discrete predictions, cascading into large waveform divergence.

## 3. Methodology

### 3.1 Why Bare-Metal: VitisAI Fragmentation

Before pursuing bare-metal programming, we evaluated AMD's recommended VitisAI auto-partitioning path. VitisAI compiles ONNX models by assigning supported operations to the NPU and falling back to CPU for unsupported operations (Gather, Shape, Unsqueeze, LSTM, Cast, etc.).

For Kokoro's 2,464-node ONNX graph, VitisAI produced 36 subgraphs with 1,080 of 2,082 operations on NPU (93% of compute GOPs). However, the 36 CPU-NPU transitions introduced sufficient DMA overhead to negate all compute gains: **RTF 0.314x on NPU versus 0.31x on CPU** — zero speedup. Model surgery (LSTM decomposition, op fusion, fragmentation reduction) produced either worse fragmentation or FlexML runtime crashes. We concluded that the auto-partitioning approach is fundamentally incompatible with Kokoro's hybrid architecture and pivoted to bare-metal programming of the ALBERT encoder.

### 3.2 Bare-Metal Implementation

We implemented the ALBERT encoder using IRON (Intermediate Representation for Operators on NPU), programming individual AIE2P tiles and DMA data movement explicitly via MLIR-AIE.

Each ALBERT pass executes the following POST-norm sequence:

```
Input [128 x 768]
  → GEMM Q (768x768) + GEMM K (768x768) + GEMM V (768x768)  [NPU]
  → Multi-Head Attention (12 heads, head_dim=64)               [CPU/NPU]
  → GEMM Attn Dense (768x768)                                  [NPU]
  → Residual Add + LayerNorm                                    [CPU]
  → GEMM FFN Up (768x2048) + GELU                              [NPU + CPU]
  → GEMM FFN Down (2048x768)                                   [NPU]
  → Residual Add + LayerNorm                                    [CPU]
  → Output [128 x 768] → fed back as next pass input
```

GEMMs use the IRON reference GEMM operator with BFP16 storage and FP32 accumulation (`prio_accuracy`), dispatched across 16-32 tiles depending on matrix dimensions. Multi-head attention uses either NPU IRON MHA (BFP16) or CPU AVX2 (FP32), depending on the configuration being tested.

Weight matrices were extracted from the official Kokoro ONNX model and converted to BF16: 4 GEMM weights at 768x768, 1 at 768x2048, and 1 at 2048x768, plus 6 bias vectors and 4 LayerNorm parameters (10.5 MB total).

### 3.3 Input Extraction and Reference Comparison

Test inputs were extracted from a real ONNX Runtime FP32 inference session using text input tokenized through Kokoro's phoneme tokenizer. This distinction is critical: random inputs produce near-uniform attention distributions that mask precision issues in softmax. Real inputs contain peaked, sparse attention patterns that are sensitive to small numerical perturbations.

We captured per-pass intermediate outputs from both the FP32 reference (ONNX Runtime on CPU) and the NPU pipeline, enabling pass-by-pass precision tracking.

### 3.4 Configuration Matrix

We tested six configurations spanning different GEMM and MHA precision combinations:

| # | GEMM | MHA | Purpose |
|---|------|-----|---------|
| 1 | BFP16 NPU | BFP16 NPU | Full NPU (fastest) |
| 2 | BFP16 NPU | Native BF16 NPU | Test BFP16 vs element-wise |
| 3 | BFP16 NPU | CPU FP32 (naive) | Isolate MHA error |
| 4 | BFP16 NPU | CPU FP32 (AVX2) | Best hybrid (fast + precise MHA) |
| 5 | Native BF16 NPU | CPU FP32 (AVX2) | Isolate BFP16 GEMM effect |
| 6 | FP32 CPU | FP32 CPU | Reference baseline |

Additionally, we tested BF16-simulated CPU execution (FP32 compute, BF16 storage between passes) to isolate storage truncation from compute truncation.

### 3.5 Audio Evaluation

The final ALBERT output (pass 11) was injected back into the full Kokoro ONNX model for end-to-end audio synthesis. Resulting waveforms were compared to the FP32 reference via Pearson correlation, signal-to-noise ratio (SNR), and duration analysis.

## 4. Results

### 4.1 Performance

**IRON GEMM achieves 248x speedup over our initial custom kernel** (0.32ms vs 79.5ms for 128x768x768), reaching 475-910 GFLOPS depending on matrix dimensions — approaching peak throughput for memory-bandwidth-bound workloads at 0.42 ops/byte arithmetic intensity.

The full ALBERT benchmark progressed through six optimization steps (Figure 6):

| Step | Change | 12-Pass Latency | vs CPU |
|------|--------|----------------|--------|
| 1 | Custom 4-tile GEMM + per-head MHA | 1022ms | 3.4x slower |
| 2 | IRON GEMM + IRON MHA | 261ms | 1.15x faster |
| 3 | Buffer reuse + fast GELU | 184ms | 1.63x faster |
| 4 | Real weights + POST-norm architecture | 246ms | 1.22x faster |
| 5 | Fused QKV GEMM | 243ms | 1.24x faster |
| 6 | AVX2 CPU MHA (precision fix) | 229ms | 1.31x faster |

The step 3-to-4 regression is expected: random weights undercount real data movement, and the correct POST-norm architecture includes an attention output dense GEMM absent from earlier estimates.

The final configuration (229ms) breaks down per-pass as shown in Figure 2: GEMM Q/K/V (4.21ms), MHA (4.40ms), GEMM Attn Dense (0.60ms), Add+LayerNorm (0.76ms), GEMM FFN Up (3.58ms), GELU (1.48ms), GEMM FFN Down (3.21ms), Add+LayerNorm (0.81ms). NPU operations account for 61% of execution time, CPU operations 39%.

All ALBERT GEMMs at these dimensions are firmly **memory-bandwidth-bound** (arithmetic intensity 0.42 ops/byte against ~51 GB/s DDR bandwidth). Doubling tile columns from 4-col to 8-col configurations yields negligible improvement, confirming the bottleneck is feeding data to tiles, not tile compute capacity.

### 4.2 Precision Degradation

Figure 3 shows the central finding: correlation vs. FP32 reference degrades across ALBERT's 12 passes at rates that vary dramatically by configuration.

**Per-pass correlation (BFP16 GEMM + BFP16 MHA — full NPU):**

| Pass | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|
| Corr | 0.999 | 0.998 | 0.995 | 0.991 | 0.986 | 0.980 | 0.972 | 0.954 | 0.959 | 0.963 | 0.968 | 0.974 |

Correlation drops steadily through pass 7 (0.954), then partially recovers. The recovery is attributed to LayerNorm re-centering the distribution after error has saturated certain dimensions. However, the final tensor-level correlation (measured across all sequence positions jointly) is 0.698 — substantially worse than the per-pass aggregate suggests.

The configuration comparison (Table 2, Figure 4) isolates three independent error sources:

| Configuration | Final Correlation | Latency | Audio |
|---|---|---|---|
| BFP16 GEMM + BFP16 MHA | 0.698 | 244ms | Destroyed |
| BFP16 GEMM + Native BF16 MHA | 0.343 | 275ms | Destroyed |
| BFP16 GEMM + CPU FP32 MHA | 0.974 | 229ms | Duration mismatch |
| Native BF16 GEMM + CPU FP32 MHA | 0.974 | 266ms | Duration mismatch |
| BF16-simulated CPU (storage only) | 0.9999 | — | Fine |
| FP32 CPU (reference) | 1.000 | ~300ms | Reference |

**MHA is the dominant error source.** Replacing only MHA with CPU FP32 (keeping BFP16 GEMM on NPU) raises correlation from 0.698 to 0.974 — an improvement larger than any GEMM-side change.

**BFP16 is paradoxically better than native BF16 for MHA.** Native BF16 MHA (element-wise truncation without block quantization) yields 0.343 correlation — far worse than BFP16 MHA's 0.698. The likely explanation is that BFP16's shared exponent preserves relative magnitudes within attention score blocks, while element-wise BF16 independently distorts each score, destroying the softmax distribution more severely.

**GEMM truncation creates an irreducible 0.974 ceiling.** Both BFP16 and native BF16 GEMM configurations yield identical 0.974 correlation when MHA runs on CPU. The error is dominated by the BF16 data format itself, not the compute-time quantization scheme. The BF16-simulated CPU test (0.9999) confirms that BF16 storage between passes is benign — the problem is specifically BF16 truncation at GEMM inputs, where truncated values are multiplied together and accumulated across 768-element dot products.

### 4.3 Audio Impact

The 0.974 correlation ceiling from Section 4.2 sounds numerically high but is functionally insufficient. Kokoro's duration predictor — a stack of 6 LSTMs consuming the ALBERT output — is highly sensitive to distributional shift.

At 0.974 correlation, the predicted phoneme durations diverge enough to produce audio of a different total length: 118,200 samples (4.92s) vs. 106,200 samples (4.42s) in the reference — an **11.3% duration mismatch** (Figure 5). This renders waveform-level metrics meaningless, as the signals are no longer temporally aligned. The resulting audio has:

- Waveform correlation: **0.01** (uncorrelated due to length mismatch)
- Signal-to-noise ratio: **-10.5 dB**
- Subjective quality: Intelligible but different prosody; unusable as a drop-in replacement

For the full-NPU configuration (0.698 correlation): audio is completely destroyed and unintelligible.

The duration predictor acts as an **error amplifier**: small continuous distributional shifts in the 768-dim hidden state are converted into discrete frame-count differences that cascade into large waveform divergence. A 2.6% average per-dimension error (implied by 0.974 correlation) is sufficient to flip discrete duration predictions.

### 4.4 INT8 Quantization

We also evaluated INT8 quantization using AMD Quark 0.11.1 with proper text-based calibration (30 real sentences tokenized through Kokoro's phoneme tokenizer). Two configurations were tested:

| Configuration | SNR (dB) | Quality |
|---|---|---|
| Full INT8 (all nodes) | -15.6 | Destroyed |
| Selective INT8 (ALBERT+predictors in FP32, decoder/vocoder in INT8) | -16.8 | Destroyed |

Both configurations produce completely unusable audio. Selective quantization is *worse* than full quantization, likely because the INT8 decoder receives FP32-precision features it was not calibrated for. This finding demonstrates that Kokoro is precision-sensitive across the **entire pipeline**, not just the iterative ALBERT encoder. The AdaIN-based decoder and HiFi-GAN vocoder are equally intolerant of quantization error.

## 5. Analysis

### 5.1 Why Iterative Architectures Amplify BF16 Error

The fundamental issue is the interaction between limited-precision arithmetic and iterative weight application. Consider a single ALBERT pass containing 6 GEMM operations. Each GEMM truncates its FP32 inputs to BF16, introducing a relative error of approximately 2^-8 per element. Over 12 passes, this yields 72 GEMM truncation events.

In a standard 12-layer transformer with independent weights, each layer projects the representation into a different subspace. The truncation error at layer i is not aligned with layer i+1's weight matrix principal components, so errors do not systematically grow. The total error accumulates as O(sqrt(72)) — approximately 8.5x the single-GEMM error.

In ALBERT, the same weight matrices are applied 12 times. Truncation errors that happen to align with the weight matrices' dominant eigenvectors are amplified at each pass. Instead of random-walk accumulation, the error follows the weight matrix's spectral structure, potentially growing as O(12) or worse for error components aligned with large singular values. This produces the correlated degradation observed in Figure 3.

The partial recovery observed after pass 7 supports this interpretation: LayerNorm re-centers and re-scales the representation at each pass, effectively projecting out some accumulated error. But LayerNorm cannot correct errors in the *direction* of the representation — only its scale and offset — so the recovery is partial.

### 5.2 The Duration Predictor as Error Amplifier

Kokoro's duration predictor converts continuous 768-dim representations into discrete per-phoneme frame counts. This discretization acts as a hard nonlinearity: a continuous error of epsilon in the hidden state can produce a discrete change of 1 frame (or more) in the predicted duration for any phoneme whose pre-rounding value is near an integer boundary.

Across a typical utterance of 20-50 phonemes, even a small probability of per-phoneme duration error accumulates into significant total duration change. Our observed 11.3% duration mismatch (12,000 samples at 24kHz) represents an average of approximately 0.5 frames/phoneme shift — consistent with a 2.6% per-dimension representation error pushing a fraction of phonemes across integer boundaries.

This has an important implication for NPU deployment: **precision requirements must be evaluated at the full pipeline level, not just at the accelerated component.** The ALBERT encoder output at 0.974 correlation appears numerically acceptable in isolation, but the downstream LSTM predictor transforms this seemingly small error into a qualitatively different output.

### 5.3 Implications Beyond Kokoro

Our findings generalize to any architecture that iteratively applies the same weights through BF16-only hardware:

- **Diffusion models** (Stable Diffusion, DALL-E): 16-50 denoising steps through a shared UNet. Each step applies the same weights to a progressively refined representation. Our results predict that BF16 truncation will compound across steps, though the noise-to-signal direction of diffusion (starting from noise, converging to signal) may partially mitigate this by decorrelating errors at early steps.

- **Universal Transformers** [3]: Iterate a shared transformer layer until convergence. Same weight-sharing pattern as ALBERT. Our results directly predict degradation at depth 12+.

- **Iterative refinement networks**: Any architecture where output is fed back as input through the same weights (iterative amortized inference, equilibrium networks, DEQ models).

Importantly, this is not AMD-specific. Any hardware with BF16-only matrix inputs (certain TPU configurations, custom ASICs) faces the same constraint. The specific correlation degradation rate will depend on model dimensions, weight conditioning, and the depth of iteration, but the fundamental mechanism — correlated error amplification through shared weights — is hardware-agnostic.

### 5.4 When BF16 NPU Inference Works Well

Our results should not be read as a blanket indictment of NPU inference. BF16 precision is adequate for:

- **Standard (non-iterative) transformers**: Each layer's independent weights decorrelate truncation errors. LIRA [6] demonstrates effective Whisper inference on XDNA2 with a standard encoder architecture.

- **Convolutional networks**: Feed-forward architectures without weight sharing. Our tests confirm that Kokoro's Conv1d decoder and HiFi-GAN vocoder tolerate BF16 in single-pass execution.

- **Shallow iteration**: 1-3 passes through shared weights may remain within tolerance for less sensitive downstream consumers.

## 6. Related Work

**NPU inference characterization.** LIRA [6] demonstrates Whisper ASR on AMD XDNA2 using VitisAI, achieving speedup with the encoder's homogeneous architecture. GPT-SoVITS on XDNA1 was reported 2-3x slower than CPU due to the same fragmentation issues we document. To our knowledge, no prior work has characterized BF16 precision through iterative architectures on bare-metal NPU hardware.

**Quantization for TTS.** INT8 quantization for TTS models has been explored primarily through VitisAI's X2 path and ONNX Runtime quantization. Our Quark results (-15.6 dB SNR) are consistent with prior observations that Kokoro is precision-sensitive, and extend them to show sensitivity across the entire pipeline.

**ALBERT precision.** The original ALBERT paper [1] does not discuss reduced-precision inference. Recent work on ALBERT for analog computing [7] explores weight-sharing on different hardware but does not characterize the precision compounding we observe.

**BF16 error analysis.** Prior work on BF16 precision focuses primarily on training stability [8], where stochastic rounding and gradient noise provide natural error decorrelation. Inference through iterative architectures lacks these corrective mechanisms, making precision degradation systematic rather than stochastic.

**Bare-metal NPU programming.** IRON [5] provides the close-to-metal programming model we use. Prior IRON work focuses on operator-level benchmarks; we extend this to a complete multi-operation workload with real model weights.

## 7. Conclusion

We have demonstrated that BF16 input truncation in NPU matrix units compounds through iterative weight-sharing architectures, producing outputs that are numerically close (0.974 correlation) but functionally unusable for sensitive downstream consumers. The error mechanism — correlated amplification through shared weight matrices — is fundamental to the interaction between limited-precision hardware and iterative computation, not a quirk of any specific model or NPU implementation.

Our bare-metal XDNA2 characterization shows that performance is achievable (1.31x CPU speedup, 229ms for 12 ALBERT passes through 4.5x optimization from initial naive implementation), but the precision ceiling is the binding constraint. The practical recommendation is to run iterative components on CPU and target NPU acceleration at single-pass stages (convolutions, standard transformer layers, vocoders).

For NPU hardware designers: adding an FP32 input path to the matrix multiply unit, or supporting mixed-precision modes where inter-pass storage remains FP32, would unlock iterative architectures without sacrificing throughput for single-pass workloads. For model designers: BF16-aware training with quantization noise injection during iterative passes could produce models that are robust to the truncation we observe, though this requires awareness of the deployment target during training.

All code, raw data, and analysis tools are available at [https://github.com/syntax1killer/npu-tts-research](https://github.com/syntax1killer/npu-tts-research).

## References

[1] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." ICLR 2020.

[2] J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

[3] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, L. Kaiser. "Universal Transformers." ICLR 2019.

[4] Hexgrad. "Kokoro TTS." https://huggingface.co/hexgrad/Kokoro-82M, 2024.

[5] AMD. "IRON: Close-to-Metal Programming for AI Engine." https://github.com/amd/IRON, 2024.

[6] AMD. "LIRA: Leveraging Intelligence with Ryzen AI." https://github.com/amd/LIRA, 2025.

[7] M. Aguirre et al. "ALBERT on Analog In-Memory Computing." Nature Communications, 2025.

[8] P. Kalamkar et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv:1905.12322, 2019.

---

*Figures referenced in this paper: Figure 1 (Architecture), Figure 2 (Timing Breakdown), Figure 3 (Precision Curve), Figure 4 (Error Waterfall), Figure 5 (Spectrograms), Figure 6 (Optimization Trajectory). See `paper/figures/` directory.*
