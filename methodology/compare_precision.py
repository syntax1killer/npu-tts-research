"""M4: Compare NPU ALBERT output vs ONNX Runtime FP32 reference.

Loads per-pass outputs from:
  - ORT FP32 reference: m4_real/X_pass_N_f32.bin
  - NPU BFP16 output:   m4_real/npu/X_pass_N.bin (bf16)

Reports per-pass error metrics: max error, mean error, RMSE, correlation.

Usage:
    python compare_precision.py [--ref-dir PATH] [--npu-dir PATH]
"""
import argparse
import os
import sys

import numpy as np


def load_f32(path, n):
    return np.fromfile(path, dtype=np.float32, count=n)


def load_bf16_as_f32(path, n):
    bf16 = np.fromfile(path, dtype=np.uint16, count=n)
    f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
    return f32


def metrics(ref, npu):
    """Compute error metrics between FP32 reference and NPU output."""
    diff = np.abs(ref - npu)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    # Pearson correlation
    if np.std(ref) > 1e-8 and np.std(npu) > 1e-8:
        corr = np.corrcoef(ref.ravel(), npu.ravel())[0, 1]
    else:
        corr = float('nan')

    # Cosine similarity
    norm_ref = np.linalg.norm(ref)
    norm_npu = np.linalg.norm(npu)
    if norm_ref > 1e-8 and norm_npu > 1e-8:
        cosine = np.dot(ref.ravel(), npu.ravel()) / (norm_ref * norm_npu)
    else:
        cosine = float('nan')

    # Signal-to-noise ratio (dB)
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')

    return {
        'max_err': max_err,
        'mean_err': mean_err,
        'rmse': rmse,
        'corr': corr,
        'cosine': cosine,
        'snr_db': snr_db,
    }


def compare(ref_dir, npu_dir):
    SEQ, DIM = 128, 768
    N = SEQ * DIM

    print("=" * 72)
    print("M4 Precision Analysis: NPU (BFP16) vs ONNX Runtime (FP32)")
    print("  Real ALBERT input from ONNX model (not random)")
    print("=" * 72)

    # Load ALBERT input (same for both)
    ref_input = load_f32(os.path.join(ref_dir, "X_init_f32.bin"), N)
    npu_input = load_bf16_as_f32(os.path.join(npu_dir, "X_init.bin"), N)
    input_diff = np.max(np.abs(ref_input - npu_input))
    print(f"\nInput quantization error (FP32->BF16): max={input_diff:.6f}")
    print(f"  Input range: [{ref_input.min():.4f}, {ref_input.max():.4f}]")

    print(f"\n{'Pass':>4} | {'MaxErr':>8} | {'MeanErr':>8} | {'RMSE':>8} | {'Corr':>8} | {'Cosine':>8} | {'SNR(dB)':>8}")
    print("-" * 72)

    all_metrics = []
    for i in range(12):
        ref_path = os.path.join(ref_dir, f"X_pass_{i}_f32.bin")
        npu_path = os.path.join(npu_dir, f"X_pass_{i}.bin")

        if not os.path.exists(ref_path):
            print(f"  Pass {i}: MISSING ref {ref_path}")
            continue
        if not os.path.exists(npu_path):
            print(f"  Pass {i}: MISSING npu {npu_path}")
            continue

        ref = load_f32(ref_path, N)
        npu = load_bf16_as_f32(npu_path, N)

        m = metrics(ref, npu)
        all_metrics.append(m)

        print(f"  {i:2d}  | {m['max_err']:8.4f} | {m['mean_err']:8.4f} | {m['rmse']:8.4f} | "
              f"{m['corr']:8.4f} | {m['cosine']:8.6f} | {m['snr_db']:8.2f}")

    if not all_metrics:
        print("\nNo data to compare!")
        return

    print("-" * 72)

    # Summary
    final = all_metrics[-1]
    print(f"\nFinal pass (pass 11) summary:")
    print(f"  Max absolute error: {final['max_err']:.4f}")
    print(f"  Mean absolute error: {final['mean_err']:.4f}")
    print(f"  RMSE: {final['rmse']:.4f}")
    print(f"  Pearson correlation: {final['corr']:.6f}")
    print(f"  Cosine similarity: {final['cosine']:.6f}")
    print(f"  SNR: {final['snr_db']:.2f} dB")

    # Error distribution for final pass
    ref = load_f32(os.path.join(ref_dir, "X_pass_11_f32.bin"), N)
    npu = load_bf16_as_f32(os.path.join(npu_dir, "X_pass_11.bin"), N)
    diff = np.abs(ref - npu)

    print(f"\n  Error distribution (final pass):")
    for threshold in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        pct = 100.0 * np.sum(diff < threshold) / len(diff)
        print(f"    < {threshold:5.2f}: {pct:6.2f}%")

    # Verdict
    print(f"\n{'=' * 72}")
    if final['corr'] > 0.99:
        print("VERDICT: EXCELLENT — NPU output highly correlated with FP32 reference")
    elif final['corr'] > 0.95:
        print("VERDICT: GOOD — NPU output well-correlated, minor precision loss")
    elif final['corr'] > 0.90:
        print("VERDICT: ACCEPTABLE — Noticeable precision loss but likely usable")
    elif final['corr'] > 0.80:
        print("VERDICT: MARGINAL — Significant precision loss, may affect audio quality")
    else:
        print("VERDICT: POOR — NPU output diverges from FP32 reference")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default=r"C:\Users\synta\npu-tts\bare-metal\kokoro\m4_real")
    parser.add_argument("--npu-dir", default=r"C:\Users\synta\npu-tts\bare-metal\kokoro\m4_real\npu")
    args = parser.parse_args()
    compare(args.ref_dir, args.npu_dir)
