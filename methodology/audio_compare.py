"""M4: Generate audio from NPU ALBERT output and compare vs FP32 reference.

Approach:
  1. Run full Kokoro model in FP32 -> reference audio
  2. Extract post-ALBERT submodel (ALBERT output -> audio)
  3. Feed NPU's ALBERT output (pass 11) through submodel -> NPU audio
  4. Compare waveforms and save both as .wav

Usage:
    python audio_compare.py [--npu-dir PATH] [--outdir PATH]
"""
import argparse
import os
import sys
import wave
import struct

import numpy as np

try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("pip install onnxruntime onnx")
    sys.exit(1)

# Path to the Kokoro ONNX model — adjust for your environment
MODEL = os.environ.get("KOKORO_MODEL", "./models/kokoro-static-128-clean.onnx")
ALBERT_OUTPUT = "/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_11/LayerNormalization_output_0"


def make_test_inputs():
    """Same inputs as extract_albert_io.py for reproducibility."""
    rng = np.random.RandomState(42)
    tokens = rng.randint(1, 150, size=(1, 128)).astype(np.int64)
    tokens[0, 95:] = 0
    style = rng.randn(1, 256).astype(np.float32) * 0.5
    speed = np.array([1.0], dtype=np.float32)
    return tokens, style, speed


def load_bf16_as_f32(path, n):
    bf16 = np.fromfile(path, dtype=np.uint16, count=n)
    return (bf16.astype(np.uint32) << 16).view(np.float32)


def save_wav(path, audio, sample_rate=24000):
    """Save float32 audio as 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def run_reference(tokens, style, speed):
    """Run full model in FP32 to get reference audio + ALBERT output."""
    print("Running full model (FP32 reference)...")

    # Load and modify model to also output ALBERT tensor
    model = onnx.load(MODEL)
    model.graph.output.append(
        helper.make_tensor_value_info(ALBERT_OUTPUT, TensorProto.FLOAT, None)
    )
    tmp = "_tmp_ref.onnx"
    onnx.save(model, tmp)
    del model

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(tmp, opts, providers=["CPUExecutionProvider"])

    input_names = [inp.name for inp in session.get_inputs()]
    feed = dict(zip(input_names, [tokens, style, speed]))

    audio_ref, albert_out_ref = session.run(["audio", ALBERT_OUTPUT], feed)
    os.remove(tmp)

    print(f"  Reference audio: {audio_ref.shape}, range [{audio_ref.min():.4f}, {audio_ref.max():.4f}]")
    print(f"  ALBERT output: {albert_out_ref.shape}")
    return audio_ref.ravel(), albert_out_ref


def build_post_albert_model():
    """Build a submodel: ALBERT output + style + speed -> audio.

    Uses onnx.utils.Extractor to extract the subgraph.
    """
    print("Extracting post-ALBERT submodel...")

    # onnx.utils.Extractor needs the model and input/output tensor names
    model = onnx.load(MODEL)

    # We need to figure out which inputs the post-ALBERT part needs.
    # It needs: ALBERT output, style, speed (style/speed used by decoder)
    # The tokens input is only used before ALBERT, so we can drop it.

    # Use Extractor to get subgraph
    from onnx.utils import Extractor
    e = Extractor(model)

    # Input: ALBERT output tensor, style, speed
    # Output: audio
    sub = e.extract_model(
        input_names=[ALBERT_OUTPUT, "tokens", "style", "speed"],
        output_names=["audio"]
    )

    tmp = "_tmp_post_albert.onnx"
    onnx.save(sub, tmp)
    del model, sub

    print(f"  Saved submodel: {tmp}")
    return tmp


def run_with_npu_albert(post_model_path, npu_albert_output, style, speed):
    """Run post-ALBERT submodel with NPU's ALBERT output."""
    print("Running post-ALBERT model with NPU output...")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(post_model_path, opts, providers=["CPUExecutionProvider"])

    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  Submodel inputs: {input_names}")

    # Also need tokens for downstream ops (e.g., duration/length references)
    rng = np.random.RandomState(42)
    tokens = rng.randint(1, 150, size=(1, 128)).astype(np.int64)
    tokens[0, 95:] = 0

    feed = {}
    for name in input_names:
        if name == ALBERT_OUTPUT:
            feed[name] = npu_albert_output
        elif name == "tokens":
            feed[name] = tokens
        elif name == "style":
            feed[name] = style
        elif name == "speed":
            feed[name] = speed
        else:
            print(f"  WARNING: unexpected input '{name}'")

    audio_npu = session.run(["audio"], feed)
    audio_npu = audio_npu[0].ravel()
    print(f"  NPU audio: {audio_npu.shape}, range [{audio_npu.min():.4f}, {audio_npu.max():.4f}]")
    return audio_npu


def compare_audio(ref, npu):
    """Compare two audio waveforms."""
    # Truncate to same length
    min_len = min(len(ref), len(npu))
    ref = ref[:min_len]
    npu = npu[:min_len]

    diff = np.abs(ref - npu)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    # Correlation
    if np.std(ref) > 1e-8 and np.std(npu) > 1e-8:
        corr = np.corrcoef(ref, npu)[0, 1]
    else:
        corr = float('nan')

    # SNR
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(diff ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    print(f"\n  Audio comparison ({min_len} samples, {min_len/24000:.2f}s):")
    print(f"    Max error:    {max_err:.6f}")
    print(f"    Mean error:   {mean_err:.6f}")
    print(f"    RMSE:         {rmse:.6f}")
    print(f"    Correlation:  {corr:.6f}")
    print(f"    SNR:          {snr_db:.2f} dB")

    return corr, snr_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to kokoro-static-128-clean.onnx (or set KOKORO_MODEL env var)")
    parser.add_argument("--npu-dir", default="./data/m4_real/npu",
                        help="Directory containing NPU per-pass BF16 outputs")
    parser.add_argument("--outdir", default="./data/m4_real",
                        help="Output directory for WAV files")
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model
    os.makedirs(args.outdir, exist_ok=True)
    tokens, style, speed = make_test_inputs()

    # 1. Reference audio (full FP32)
    audio_ref, albert_ref = run_reference(tokens, style, speed)

    # 2. Load NPU ALBERT output (pass 11, bf16)
    npu_path = os.path.join(args.npu_dir, "X_pass_11.bin")
    npu_albert = load_bf16_as_f32(npu_path, 128 * 768)
    npu_albert = npu_albert.reshape(1, 128, 768)  # add batch dim
    print(f"\nNPU ALBERT output: shape={npu_albert.shape}, "
          f"range=[{npu_albert.min():.4f}, {npu_albert.max():.4f}]")

    # Quick check: ALBERT output correlation
    albert_ref_2d = albert_ref.squeeze(0).ravel()
    npu_albert_2d = npu_albert.squeeze(0).ravel()
    albert_corr = np.corrcoef(albert_ref_2d, npu_albert_2d)[0, 1]
    print(f"ALBERT output correlation (ref vs NPU): {albert_corr:.6f}")

    # 3. Build post-ALBERT submodel
    post_model = build_post_albert_model()

    # 4. Run post-ALBERT with NPU output
    audio_npu = run_with_npu_albert(post_model, npu_albert, style, speed)

    # Cleanup temp model
    if os.path.exists(post_model):
        os.remove(post_model)

    # 5. Compare
    corr, snr = compare_audio(audio_ref, audio_npu)

    # 6. Save WAVs
    ref_wav = os.path.join(args.outdir, "audio_ref_fp32.wav")
    npu_wav = os.path.join(args.outdir, "audio_npu_bf16.wav")
    save_wav(ref_wav, audio_ref)
    save_wav(npu_wav, audio_npu)
    print(f"\n  Saved: {ref_wav}")
    print(f"  Saved: {npu_wav}")

    # Verdict
    print(f"\n{'=' * 60}")
    if corr > 0.99:
        print("AUDIO VERDICT: EXCELLENT - Indistinguishable from reference")
    elif corr > 0.95:
        print("AUDIO VERDICT: GOOD - Minor differences, likely inaudible")
    elif corr > 0.90:
        print("AUDIO VERDICT: ACCEPTABLE - Some differences, listen to judge")
    elif corr > 0.80:
        print("AUDIO VERDICT: MARGINAL - Noticeable differences expected")
    else:
        print("AUDIO VERDICT: POOR - Significant audio degradation expected")
    print(f"  Correlation: {corr:.6f}, SNR: {snr:.2f} dB")
    print(f"  Listen and compare: {ref_wav}")
    print(f"                  vs: {npu_wav}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
