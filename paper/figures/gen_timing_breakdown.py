"""
Figure 2: Per-pass timing breakdown of ALBERT on NPU.
Horizontal stacked bar + pie chart for device split.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

operations = [
    ("GEMM Q/K/V",      4.21, "NPU"),
    ("MHA (12 heads)",   4.40, "CPU"),
    ("GEMM Attn Dense",  0.60, "NPU"),
    ("Add+LN",           0.76, "CPU"),
    ("GEMM FFN Up",      3.58, "NPU"),
    ("GELU",             1.48, "CPU"),
    ("GEMM FFN Down",    3.21, "NPU"),
    ("Add+LN",           0.81, "CPU"),
]

npu_color = "#2196F3"
cpu_color = "#FF9800"
colors = [npu_color if d == "NPU" else cpu_color for _, _, d in operations]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2.2, 1]})

# Left: stacked horizontal bar
bottom = 0
for i, (label, time, device) in enumerate(operations):
    ax1.barh(0, time, left=bottom, color=colors[i], edgecolor='white', linewidth=0.8, height=0.6)
    # Only label inside bars wide enough (>2ms)
    if time >= 2.0:
        ax1.text(bottom + time/2, 0, f"{label}\n{time:.2f}ms",
                 ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    bottom += time

# Labels for narrow bars go below with leader lines
narrow_ops = [(i, op) for i, op in enumerate(operations) if op[1] < 2.0]
# Calculate positions of each bar
positions = []
cum = 0
for _, t, _ in operations:
    positions.append(cum + t/2)
    cum += t

y_label = -0.55
for idx, (i, (label, time, device)) in enumerate(narrow_ops):
    x_center = positions[i]
    # Stagger labels to avoid overlap
    offsets = [-0.8, 0.8, -0.8, 0.8]
    x_offset = offsets[idx % len(offsets)] if len(narrow_ops) > 1 else 0
    x_target = x_center + x_offset
    ax1.annotate(f"{label}\n{time:.2f}ms",
                 xy=(x_center, -0.3), xytext=(x_target, y_label - 0.15 * (idx % 2)),
                 ha='center', va='top', fontsize=7, color='#333',
                 arrowprops=dict(arrowstyle='-', color='#999', lw=0.8))

total = sum(t for _, t, _ in operations)
ax1.set_xlim(0, total * 1.02)
ax1.set_ylim(-1.3, 0.6)
ax1.set_xlabel('Time (ms)', fontsize=11)
ax1.set_yticks([])
ax1.set_title(f'Single ALBERT Pass: {total:.1f}ms', fontsize=12, pad=10)

npu_ms = sum(t for _, t, d in operations if d == "NPU")
cpu_ms = sum(t for _, t, d in operations if d == "CPU")
npu_patch = mpatches.Patch(color=npu_color, label=f'NPU  {npu_ms:.1f}ms ({npu_ms/total*100:.0f}%)')
cpu_patch = mpatches.Patch(color=cpu_color, label=f'CPU  {cpu_ms:.1f}ms ({cpu_ms/total*100:.0f}%)')
ax1.legend(handles=[npu_patch, cpu_patch], loc='upper right', fontsize=9, framealpha=0.95)

# Right: pie chart with realistic overhead
npu_total = npu_ms * 12
cpu_total = cpu_ms * 12
total_measured = 229.0
overhead = total_measured - npu_total - cpu_total

# Only include overhead slice if meaningful (>1ms)
if overhead > 1:
    sizes = [npu_total, cpu_total, overhead]
    pie_labels = [f'NPU\n{npu_total:.0f}ms', f'CPU\n{cpu_total:.0f}ms', f'Sync\n{overhead:.0f}ms']
    pie_colors = [npu_color, cpu_color, '#BDBDBD']
else:
    sizes = [npu_total, cpu_total]
    pie_labels = [f'NPU\n{npu_total:.0f}ms', f'CPU\n{cpu_total:.0f}ms']
    pie_colors = [npu_color, cpu_color]

wedges, texts = ax2.pie(sizes, labels=pie_labels, colors=pie_colors, startangle=90,
                         textprops={'fontsize': 9}, wedgeprops=dict(edgecolor='white', linewidth=1))

# Add percentages
for i, (size, wedge) in enumerate(zip(sizes, wedges)):
    pct = size / total_measured * 100
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.55 * np.cos(np.radians(angle))
    y = 0.55 * np.sin(np.radians(angle))
    ax2.text(x, y, f'{pct:.0f}%', ha='center', va='center', fontsize=10,
             fontweight='bold', color='white')

ax2.set_title(f'Full ALBERT: 12 passes, {total_measured:.0f}ms', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_timing_breakdown.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig2_timing_breakdown.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig2_timing_breakdown.png'}")
