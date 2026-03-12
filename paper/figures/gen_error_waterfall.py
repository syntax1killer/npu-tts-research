"""
Figure 4: Error source waterfall — isolating contributions from
storage truncation, GEMM input truncation, and MHA truncation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Data: each stage and its resulting correlation
sources = [
    ("FP32\nReference",           1.000,  "#66BB6A"),
    ("After BF16\nStorage",       0.9999, "#FFCA28"),
    ("After GEMM\nTruncation\n(72 events)", 0.974, "#FF7043"),
    ("After MHA\nTruncation\n(12 events)",  0.698, "#EF5350"),
]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(sources))
bar_width = 0.55

for i, (label, corr, color) in enumerate(sources):
    # Solid bar = remaining correlation
    ax.bar(i, corr, bar_width, color=color, edgecolor='#333', linewidth=0.8, zorder=3)

    # Hatched "lost" portion on top
    if i > 0:
        prev_corr = sources[i-1][1]
        drop = prev_corr - corr
        ax.bar(i, drop, bar_width, bottom=corr, color=color, edgecolor='#333',
               linewidth=0.8, alpha=0.25, hatch='///', zorder=3)
        # Connector line from previous bar
        ax.plot([i - 1 + bar_width/2, i - bar_width/2], [prev_corr, prev_corr],
                'k--', linewidth=0.8, alpha=0.4, zorder=2)
        # Drop label — outside the bar to avoid readability issues
        if drop < 0.005:
            ax.text(i + bar_width/2 + 0.08, prev_corr - drop/2,
                    f'  -{drop:.4f}', ha='left', va='center', fontsize=9,
                    color='#333', fontweight='bold')
        else:
            ax.text(i + bar_width/2 + 0.08, corr + drop/2,
                    f'  -{drop:.3f}', ha='left', va='center', fontsize=9,
                    color='#333', fontweight='bold')

    # Correlation value label above bar
    fmt = f'{corr:.4f}' if corr > 0.99 else f'{corr:.3f}'
    ax.text(i, corr + 0.012, fmt, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Threshold annotation
ax.axhline(y=0.974, color='#2ca02c', linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in sources], fontsize=9.5)
ax.set_ylabel('Pearson Correlation vs FP32 Reference', fontsize=11)
ax.set_title('Error Source Isolation: Where Precision Is Lost\n'
             'ALBERT 12-pass final output on AMD XDNA2 NPU', fontsize=12)
ax.set_ylim(0.6, 1.06)
ax.set_xlim(-0.5, 3.8)
ax.grid(True, axis='y', alpha=0.2, zorder=0)

# Legend-style annotation
ax.text(0.97, 0.06, 'Hatched = precision lost at each stage\n'
        'Dominant source: MHA BFP16 (80% of total loss)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#FFB74D', linewidth=1))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_error_waterfall.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig4_error_waterfall.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig4_error_waterfall.png'}")
