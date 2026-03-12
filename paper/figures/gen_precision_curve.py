"""
Figure 3: Per-pass precision degradation across ALBERT's 12 iterative passes.
KEY FIGURE — shows BF16 error compounding through iterative weight-sharing.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\Users\synta\npu-tts\bare-metal\kokoro\m4_real")
OUT_DIR = Path(__file__).resolve().parent
SEQ_LEN, DIM = 128, 768

def load_pass(directory, pass_num):
    """Load a BF16 pass output and convert to FP32."""
    path = directory / f"X_pass_{pass_num}.bin"
    raw = np.fromfile(path, dtype=np.uint16)
    fp32_bits = raw.astype(np.uint32) << 16
    return np.frombuffer(fp32_bits.tobytes(), dtype=np.float32).reshape(SEQ_LEN, DIM)

def pearson_corr(a, b):
    a_flat, b_flat = a.flatten(), b.flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    num = np.dot(a_c, b_c)
    den = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
    return num / den if den > 0 else 0.0

configs = [
    ("npu",                     "BFP16 GEMM + BFP16 MHA (full NPU)",     "#d62728", "o", "-"),
    ("npu_nobfp16",             "BFP16 GEMM + native BF16 MHA",          "#ff7f0e", "s", "-"),
    ("npu_cpumha_avx2",         "BFP16 GEMM + CPU FP32 MHA (best)",      "#2ca02c", "^", "-"),
    ("npu_nobfp16gemm_cpumha",  "Native BF16 GEMM + CPU FP32 MHA",       "#1f77b4", "D", "--"),
]

passes = list(range(12))

fig, ax = plt.subplots(figsize=(10, 6.5))

for dirname, label, color, marker, ls in configs:
    corrs = []
    config_dir = DATA_DIR / dirname
    if not config_dir.exists():
        print(f"Skipping {dirname}")
        continue
    for p in passes:
        ref = np.fromfile(DATA_DIR / f"X_pass_{p}_f32.bin", dtype=np.float32).reshape(SEQ_LEN, DIM)
        npu = load_pass(config_dir, p)
        corrs.append(pearson_corr(ref, npu))
    ax.plot(passes, corrs, marker=marker, linestyle=ls, label=label,
            color=color, linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.8)

# BF16 storage-only reference line
ax.axhline(y=0.9999, color='#888', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(11.4, 0.998, 'BF16 storage only\n(FP32 compute): 0.9999', fontsize=8,
        color='#666', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc'))

# Threshold annotation — dark background for readability
ax.axhspan(0.965, 0.983, color='#2ca02c', alpha=0.08)
ax.text(0.2, 0.957, 'Best NPU: 0.974 — duration predictor fails at this level',
        fontsize=9, color='#1b5e20', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2ca02c', alpha=0.9))

ax.set_xlabel('ALBERT Pass Number', fontsize=12)
ax.set_ylabel('Pearson Correlation vs FP32 Reference', fontsize=12)
ax.set_title('BF16 Precision Degradation Through Iterative Weight-Sharing\n'
             'Kokoro TTS ALBERT Encoder on AMD XDNA2 NPU', fontsize=13)
ax.set_xticks(passes)
ax.set_ylim(0.25, 1.03)
ax.set_xlim(-0.3, 11.5)
ax.legend(loc='lower left', fontsize=9, framealpha=0.95, edgecolor='#ccc')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_precision_curve.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig3_precision_curve.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig3_precision_curve.png'}")
