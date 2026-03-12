"""
Figure 6: Optimization trajectory from naive NPU to final configuration.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

steps = [
    ("Custom GEMM\n+ Phase 3 MHA", 1022),
    ("IRON GEMM\n+ IRON MHA",      261),
    ("Buffer reuse\n+ fast GELU",   184),
    ("Real weights\n+ POST-norm",   246),
    ("Fused QKV\nGEMM",            243),
    ("AVX2 CPU MHA\n(precision)",   229),
]

labels = [s[0] for s in steps]
times = [s[1] for s in steps]
cpu_baseline = 300

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(steps))
bar_colors = ['#ef5350' if t > cpu_baseline else '#66bb6a' for t in times]
bars = ax.bar(x, times, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6, zorder=3)

# CPU baseline
ax.axhline(y=cpu_baseline, color='#1565C0', linestyle='--', linewidth=2,
           label=f'CPU FP32 baseline ({cpu_baseline}ms)', zorder=2)

# Labels on bars — above for tall, inside top for short
for i, (label, time) in enumerate(steps):
    if time > cpu_baseline:
        text = f'{time}ms\n{time/cpu_baseline:.1f}x slower'
        ax.text(i, time + 15, text, ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color='#b71c1c')
    else:
        speedup = cpu_baseline / time
        ax.text(i, time + 15, f'{time}ms', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#1b5e20')
        ax.text(i, time - 15, f'{speedup:.2f}x', ha='center', va='top', fontsize=8,
                color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('ALBERT 12-Pass Latency (ms)', fontsize=11)
ax.set_title('Optimization Trajectory: 4.5x Improvement Through Bare-Metal Engineering', fontsize=12)
ax.set_ylim(0, 1120)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, axis='y', alpha=0.2, zorder=0)

# Regression annotation — positioned to not overlap bars
ax.annotate('Regression expected:\nrandom weights understated\nreal data movement',
            xy=(3, 260), xytext=(4.3, 600),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.2),
            fontsize=8, color='#666', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ccc'))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_optimization_trajectory.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig6_optimization_trajectory.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig6_optimization_trajectory.png'}")
