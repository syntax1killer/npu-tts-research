# BF16 Precision Limits in Iterative Neural Architectures on AMD XDNA2 NPU
## Paper Outline (targeting arXiv preprint)

---

## Abstract (~150 words)
- Neural Processing Units (NPUs) promise efficient on-device AI inference via low-precision matrix units
- AMD XDNA2's AIE2P tiles only accept BF16 inputs to aie::mmul (no FP32 path)
- We perform the first bare-metal characterization of BF16 precision through an iterative weight-sharing architecture (ALBERT in Kokoro TTS) on AMD XDNA2 hardware
- Key finding: BF16 input truncation compounds through 12 iterative passes via correlated error amplification, producing 0.974 correlation (best case) — insufficient for downstream duration prediction (unusable audio)
- Contrast with single-pass architectures where BF16 is adequate
- Performance is viable (229ms vs 300ms CPU, 1.31x speedup) — precision is the blocker
- Implications for deploying diffusion, Universal Transformers, and other iterative models on BF16-only hardware

---

## 1. Introduction (~1.5 pages)
- NPU proliferation in consumer hardware (AMD XDNA2, Intel Meteor Lake, Qualcomm Hexagon)
- Promise: efficient, low-power AI inference for always-on workloads
- Reality: BF16 matrix units with no FP32 input path create precision constraints
- This paper: first public bare-metal characterization of BF16 precision through iterative weight-sharing on real NPU hardware
- Contribution 1: Empirical demonstration that BF16 truncation compounds through weight-sharing loops
- Contribution 2: Isolation of error sources (GEMM input truncation vs accumulation vs storage)
- Contribution 3: Performance characterization of bare-metal IRON pipeline on XDNA2
- Contribution 4: Practical guidelines for NPU deployment of iterative architectures

## 2. Background (~1.5 pages)

### 2.1 AMD XDNA2 Architecture
- AIE2P tile array (4x8), aie::mmul<4,8,8> BF16 intrinsic
- FP32 accumulation via prio_accuracy mode
- BFP16 block floating point (shared exponent per block)
- Memory hierarchy: L1 (64KB per tile) → L2 (shared) → DDR
- XRT runtime, xclbin kernel dispatch

### 2.2 BF16 Numerical Properties
- 8-bit exponent, 7-bit mantissa (~3.9 decimal digits)
- Per-element truncation error: ~2^-8 relative
- Contrast with FP16 (5-bit exponent, 10-bit mantissa) — different trade-off
- BFP16 variant: shared exponent per block

### 2.3 ALBERT Architecture
- Weight sharing: single transformer layer applied N times iteratively
- Reduces parameters (shared weights) at cost of iterative computation
- Each pass: 6 GEMMs + 1 MHA + 2 LayerNorms + GELU + residual connections
- Used in Kokoro TTS as text encoder (12 passes, dim=768)

### 2.4 Kokoro TTS Pipeline
- Encoder (ALBERT) → Duration Predictor (6 LSTMs) → Decoder (Conv+AdaIN) → Vocoder (HiFi-GAN)
- Duration predictor is the critical downstream consumer of ALBERT output

## 3. Methodology (~1.5 pages)

### 3.1 Bare-Metal NPU Programming
- MLIR-AIE / IRON flow (bypassing VitisAI auto-partitioning)
- Why bare-metal: VitisAI produces 36 subgraphs with CPU↔NPU fragmentation, zero speedup
- Build pipeline: WSL2 (MLIR → xclbin) → Windows (MSVC host) → XRT

### 3.2 Benchmark Design
- Real model weights extracted from ONNX (16 binary files, 10.5 MB)
- Real input tensors from ONNX Runtime inference (not random data — critical distinction)
- Per-pass output capture for correlation tracking
- End-to-end audio evaluation (inject ALBERT output back into full model)

### 3.3 Configuration Matrix
- 6 configurations testing GEMM precision × MHA precision combinations
- CPU FP32 reference baseline
- BF16-simulated CPU test (storage-only truncation)

### 3.4 Metrics
- Pearson correlation vs FP32 reference (per-pass and final)
- Audio SNR (dB), waveform correlation
- Duration predictor output comparison (phoneme timing divergence)

## 4. Results (~3 pages)

### 4.1 Performance (Figure 1, Figure 2, Table 1)
- IRON GEMM: 248x speedup over naive kernel (0.32ms vs 79.5ms)
- Full ALBERT: 229ms (12 passes), 1.31x faster than CPU 300ms
- Per-pass breakdown: 8 operations, NPU 58%, CPU 37%, overhead 5%
- Optimization trajectory: 1022ms → 229ms through 6 engineering steps
- Bandwidth-bound regime: 0.42 ops/byte, DDR bottleneck

### 4.2 Precision Degradation (Figure 3, Figure 4, Table 2)
- Per-pass correlation curve: 0.9993 → 0.9541 → 0.9736 (partial LayerNorm recovery)
- Configuration comparison matrix (6 configs)
- Error source isolation: MHA dominant, GEMM residual, storage benign
- BFP16 paradox: block quantization better than element-wise for MHA

### 4.3 Audio Impact (Figure 5, Table 3)
- Duration predictor sensitivity: 120K vs 106K samples at 0.974 corr
- Audio SNR: -10.5 dB (best NPU config), 0.01 waveform correlation
- Spectrograms: FP32 vs BF16 output comparison
- INT8 quantization results: -15.6 to -16.8 dB SNR (worse than BF16)

## 5. Analysis (~1.5 pages)

### 5.1 Why Iterative Architectures Amplify BF16 Error
- Correlated error modes: same weights amplify same truncation artifacts
- Contrast with independent-layer transformers (uncorrelated errors)
- Mathematical intuition: error eigenvectors aligned with weight matrix principal components

### 5.2 The Duration Predictor as Error Amplifier
- LSTMs are highly sensitive to distributional shift in input
- 0.974 correlation ≈ 2.6% average per-dimension error → discrete duration changes → cascading audio divergence
- Non-linear amplification: small continuous error → large discrete output change

### 5.3 Implications Beyond ALBERT
- Diffusion models: 16-32 denoising steps through shared UNet = same compounding
- Universal Transformers: iterative refinement with shared weights
- Iterative refinement networks: any architecture with feedback through BF16 compute
- This is NOT an AMD-specific issue: any BF16-only matrix unit (some TPU configs, custom ASICs) faces the same constraint

### 5.4 When BF16 NPU Inference Works Well
- Standard (non-iterative) transformers: independent weights decorrelate errors
- CNNs: feed-forward, no feedback loop
- Single-pass architectures: error bounded per-layer, no compounding

## 6. Related Work (~1 page)
- AMD LIRA (Whisper on XDNA2) — homogeneous encoder, single-pass, works well
- GPT-SoVITS on XDNA1 — 2-3x slower than CPU (fragmentation, same partitioning issue)
- ALBERT on analog chip (Nature 2025) — different hardware, same architecture
- Quantization-aware training for iterative models
- BF16 error analysis in training (different context — training vs inference)
- VitisAI auto-partitioning limitations

## 7. Conclusion (~0.5 page)
- BF16 input truncation in hardware matrix units is fundamentally incompatible with deep iterative weight-sharing (12+ passes)
- Performance is achievable (1.31x CPU speedup) but precision ceiling makes output unusable
- Practical recommendation: run iterative components on CPU, target NPU at single-pass stages
- Future hardware: FP32 mmul inputs or mixed-precision modes would resolve this
- Future models: BF16-aware training with quantization noise injection during iterative passes

---

## Figures Plan

### Figure 1: ALBERT Architecture & NPU/CPU Split
- Block diagram showing the 12-pass loop
- Color-coded: green=NPU (GEMMs), blue=CPU (MHA, LayerNorm, GELU)
- Arrows showing the feedback loop where error compounds

### Figure 2: Per-Pass Timing Breakdown
- Stacked bar chart: 8 operations per pass
- Color by device (NPU vs CPU)
- Shows the 20.32ms decomposition

### Figure 3: Precision Degradation Curve (KEY FIGURE)
- X-axis: pass number (0-11)
- Y-axis: correlation vs FP32 reference
- Multiple lines for each configuration (BFP16+BFP16, BFP16+CPU, etc.)
- Highlight the 0.974 ceiling and the "usable" threshold
- Show that storage-only BF16 (0.9999) does NOT compound

### Figure 4: Error Source Waterfall
- Waterfall chart: FP32 baseline (1.0) → storage truncation → GEMM truncation → MHA truncation → final
- Quantifies each error source's contribution

### Figure 5: Audio Spectrograms
- Side-by-side: FP32 reference vs best NPU config
- Show duration mismatch visually
- Highlight where phoneme boundaries diverge

### Figure 6: Optimization Trajectory
- Line chart: latency vs optimization step (1022ms → 229ms)
- Annotated with what changed at each step

### Table 1: IRON GEMM Microbenchmarks
### Table 2: Configuration × Precision Matrix
### Table 3: INT8 Quantization Results
