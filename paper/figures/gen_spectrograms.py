"""
Figure 5: Spectrogram comparison — FP32 reference vs best NPU configuration.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
# Path to directory containing audio_ref_fp32.wav and audio_npugemm_cpumha.wav
# Adjust for your environment or set NPU_DATA_DIR env var
import os
DATA_DIR = Path(os.environ.get("NPU_DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data" / "m4_real")))

def load_wav(path):
    import wave
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        if w.getsampwidth() == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif w.getsampwidth() == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.float32)
    return sr, samples

ref_path = DATA_DIR / "audio_ref_fp32.wav"
npu_path = DATA_DIR / "audio_npugemm_cpumha.wav"

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

sr_ref, ref_samples = load_wav(ref_path)
sr_npu, npu_samples = load_wav(npu_path)

# Spectrograms
for ax, samples, sr, title in [
    (axes[0], ref_samples, sr_ref,
     f'FP32 Reference ({len(ref_samples):,} samples, {len(ref_samples)/sr_ref:.2f}s)'),
    (axes[1], npu_samples, sr_npu,
     f'NPU BFP16 GEMM + CPU FP32 MHA ({len(npu_samples):,} samples, {len(npu_samples)/sr_npu:.2f}s)'),
]:
    ax.specgram(samples, NFFT=1024, Fs=sr, noverlap=768, cmap='magma',
                scale='dB', vmin=-80, vmax=0)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title(title, fontsize=11, pad=5)
    ax.set_ylim(0, 8000)

# Waveform overlay
t_ref = np.arange(len(ref_samples)) / sr_ref
t_npu = np.arange(len(npu_samples)) / sr_npu
axes[2].plot(t_ref, ref_samples, alpha=0.6, linewidth=0.3, label='FP32 Reference', color='#1f77b4')
axes[2].plot(t_npu, npu_samples, alpha=0.6, linewidth=0.3, label='NPU Output', color='#d62728')
axes[2].set_xlabel('Time (s)', fontsize=10)
axes[2].set_ylabel('Amplitude', fontsize=10)
axes[2].set_title('Waveform Comparison: Duration Mismatch from ALBERT Precision Loss', fontsize=11, pad=5)
axes[2].legend(fontsize=10, loc='upper left', framealpha=0.9)

# Duration divergence annotation
len_diff = abs(len(ref_samples) - len(npu_samples))
longer = "NPU" if len(npu_samples) > len(ref_samples) else "FP32"
axes[2].text(0.98, 0.92,
    f'Duration divergence: {len_diff:,} samples ({len_diff/sr_ref*1000:.0f}ms)\n'
    f'{longer} output is {len_diff/min(len(ref_samples), len(npu_samples))*100:.1f}% longer\n'
    f'Cause: BF16 error in ALBERT shifts duration predictor output',
    transform=axes[2].transAxes, ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825',
              linewidth=1.2, alpha=0.95))

fig.suptitle('Audio Quality Impact of BF16 Precision Loss in ALBERT Encoder',
             fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_spectrograms.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig5_spectrograms.pdf', bbox_inches='tight')
print(f"Saved to {OUT_DIR / 'fig5_spectrograms.png'}")
