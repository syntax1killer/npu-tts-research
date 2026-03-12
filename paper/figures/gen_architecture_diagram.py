"""
Figure 1: ALBERT architecture diagram with NPU/CPU device mapping.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')

npu_color = '#2196F3'
cpu_color = '#FF9800'

def add_block(ax, x, y, w, h, label, color, sublabel=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#333', linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.1, label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.12, sublabel, ha='center', va='center',
                fontsize=7.5, color='#eee', style='italic')
    else:
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Title
ax.text(5.0, 8.2, 'Kokoro TTS: ALBERT Encoder on AMD XDNA2 NPU',
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5.0, 7.85, 'Single shared transformer layer applied 12 times iteratively',
        ha='center', va='center', fontsize=10, color='#555')

# Input block (left side)
inp_box = FancyBboxPatch((0.3, 4.5), 1.8, 0.7, boxstyle="round,pad=0.08",
                          facecolor='#E0E0E0', edgecolor='#333', linewidth=1.2)
ax.add_patch(inp_box)
ax.text(1.2, 4.85, 'Input\n[128 x 768]', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow from input to pass box
ax.annotate('', xy=(2.7, 4.85), xytext=(2.1, 4.85),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Pass box (dashed border)
pass_box = FancyBboxPatch((2.7, 0.3), 4.8, 7.0, boxstyle="round,pad=0.15",
                           facecolor='#F8F8F8', edgecolor='#888', linewidth=2,
                           linestyle='--')
ax.add_patch(pass_box)
ax.text(5.1, 7.05, 'x 12 passes (shared weights)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#888', linewidth=1.2))

# Operations inside the pass (top to bottom)
x_left = 3.0
w = 4.2
h = 0.58
gap = 0.2
y = 6.3

ops = [
    ("GEMM Q/K/V",          "3x [768x768]",        npu_color, "4.21ms"),
    ("Multi-Head Attention", "12 heads, dim=64",     cpu_color, "4.40ms"),
    ("GEMM Attn Dense",     "[768x768]",            npu_color, "0.60ms"),
    ("Add + LayerNorm",      "residual connection",  cpu_color, "0.76ms"),
    ("GEMM FFN Up + GELU",  "[768x2048]",           npu_color, "3.58ms + 1.48ms"),
    ("GEMM FFN Down",       "[2048x768]",           npu_color, "3.21ms"),
    ("Add + LayerNorm",      "residual connection",  cpu_color, "0.81ms"),
]

for i, (label, sublabel, color, timing) in enumerate(ops):
    add_block(ax, x_left, y - h, w, h, label, color, sublabel)
    # Timing label on the right
    ax.text(x_left + w + 0.15, y - h/2, timing, ha='left', va='center',
            fontsize=8, color='#444', family='monospace')
    # Small arrow between ops
    if i < len(ops) - 1:
        ax.annotate('', xy=(x_left + w/2, y - h - 0.02),
                    xytext=(x_left + w/2, y - h - gap + 0.02),
                    arrowprops=dict(arrowstyle='->', lw=1, color='#aaa'))
    y -= (h + gap)

# Output block (right side)
out_box = FancyBboxPatch((8.0, 2.0), 1.8, 0.7, boxstyle="round,pad=0.08",
                          facecolor='#E0E0E0', edgecolor='#333', linewidth=1.2)
ax.add_patch(out_box)
ax.text(8.9, 2.35, 'Output\n[128 x 768]', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow from last op to output
ax.annotate('', xy=(8.0, 2.35), xytext=(7.2, y + h + gap - 0.05),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333',
                           connectionstyle='arc3,rad=-0.2'))

# Feedback arrow (the key path — big red curve on the left)
ax.annotate('', xy=(2.85, 6.5), xytext=(2.85, 0.7),
            arrowprops=dict(arrowstyle='->', lw=3, color='#d62728',
                           connectionstyle='arc3,rad=-0.4'))

# Feedback label — positioned clear of other elements
ax.text(0.9, 3.5, 'FEEDBACK\nLOOP', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#d62728',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#d62728', linewidth=1.5))
ax.text(0.9, 2.6, 'BF16 truncation\ncompounds here', ha='center', va='center',
        fontsize=8.5, color='#d62728', style='italic')

# Legend
npu_patch = mpatches.Patch(color=npu_color, alpha=0.9, label='NPU (BF16 GEMM)')
cpu_patch = mpatches.Patch(color=cpu_color, alpha=0.9, label='CPU (FP32)')
ax.legend(handles=[npu_patch, cpu_patch], loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor='#ccc', bbox_to_anchor=(0.98, 0.01))

# Summary box
ax.text(8.9, 1.2, 'Per pass: 19.1ms\n12 passes: 229ms\nCPU baseline: 300ms\nSpeedup: 1.31x',
        ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#66bb6a', linewidth=1.2))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig1_architecture.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig1_architecture.png'}")
