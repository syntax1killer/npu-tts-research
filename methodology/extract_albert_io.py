"""M4: Extract real ALBERT input/output from ONNX Runtime for precision validation.

Runs the Kokoro model in FP32 via ORT, extracts:
  - ALBERT input tensor (after embedding projection, before first pass)
  - Per-pass output tensors (after each of the 12 LayerNorm outputs)

Saves as bf16 .bin files compatible with albert_bench.exe --dump-dir format,
and as f32 reference for comparison with NPU output.

Usage:
    python extract_albert_io.py [--outdir PATH]
"""
import argparse
import os
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("pip install onnx")
    sys.exit(1)

# Path to the Kokoro ONNX model — adjust for your environment
MODEL = os.environ.get("KOKORO_MODEL", r"C:\Users\synta\npu-tts\models\kokoro-static-128-clean.onnx")

# Tensor names from graph inspection
ALBERT_INPUT = "/encoder/bert/encoder/embedding_hidden_mapping_in/Add_output_0"

# Per-pass outputs (pass 0 input = ALBERT_INPUT, pass N output = full_layer_layer_norm_N-1)
PASS_OUTPUTS = [
    # Pass 0 output
    "/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization_output_0",
]
for i in range(1, 12):
    suffix = f"_{i}" if i > 0 else ""
    PASS_OUTPUTS.append(
        f"/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm{suffix}/LayerNormalization_output_0"
    )

# Fix: pass 0 has no suffix, passes 1-11 have _1 through _11
PASS_OUTPUTS = []
for i in range(12):
    if i == 0:
        name = "/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm/LayerNormalization_output_0"
    else:
        name = f"/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/full_layer_layer_norm_{i}/LayerNormalization_output_0"
    PASS_OUTPUTS.append(name)


def f32_to_bf16(arr):
    """FP32 -> BF16 (uint16) with round-to-nearest-even."""
    u32 = arr.astype(np.float32).ravel().view(np.uint32)
    rb = ((u32 >> 16) & 1) + 0x7FFF
    bf16 = ((u32 + rb) >> 16).astype(np.uint16)
    return bf16.reshape(arr.shape)


def make_test_inputs():
    """Create test inputs for the Kokoro model.

    tokens: [1, 128] int64 — phoneme token IDs
    style:  [1, 256] float32 — style embedding
    speed:  [1] float32 — speed factor
    """
    # Use a mix of common phoneme token IDs (not all zeros, not all same)
    # Kokoro tokenizer uses values roughly in range 0-178
    rng = np.random.RandomState(42)
    tokens = rng.randint(1, 150, size=(1, 128)).astype(np.int64)
    # Pad last ~30 tokens with 0 (padding) to simulate shorter sentence
    tokens[0, 95:] = 0

    # Style embedding — use small random values (typical magnitude ~0.1-1.0)
    style = rng.randn(1, 256).astype(np.float32) * 0.5

    # Normal speed
    speed = np.array([1.0], dtype=np.float32)

    return tokens, style, speed


def extract(outdir):
    os.makedirs(outdir, exist_ok=True)

    # Add intermediate outputs to the model
    print("Loading ONNX model for graph modification...")
    model = onnx.load(MODEL)

    # Collect all tensor names we want to extract
    intermediate_names = [ALBERT_INPUT] + PASS_OUTPUTS
    print(f"  Will extract {len(intermediate_names)} intermediate tensors")

    # Add intermediate tensors as model outputs
    # Need to find their shapes from value_info or infer
    for name in intermediate_names:
        # Create output without shape info (ORT will infer)
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )

    # Save modified model to temp file
    tmp_model = os.path.join(outdir, "_tmp_model.onnx")
    onnx.save(model, tmp_model)
    del model  # free memory

    print("Creating ORT session...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(tmp_model, opts, providers=["CPUExecutionProvider"])

    # Create inputs
    tokens, style, speed = make_test_inputs()
    print(f"  tokens: {tokens.shape} (non-zero: {np.count_nonzero(tokens)})")
    print(f"  style: {style.shape} (mean={style.mean():.3f}, std={style.std():.3f})")
    print(f"  speed: {speed}")

    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  Input names: {input_names}")
    feed = dict(zip(input_names, [tokens, style, speed]))

    # Run with all outputs
    output_names = [ALBERT_INPUT] + PASS_OUTPUTS + ["audio"]
    print(f"\nRunning inference (FP32, {len(output_names)} outputs)...")
    results = session.run(output_names, feed)

    # Extract results
    albert_input = results[0]  # ALBERT input
    pass_outputs = results[1:13]  # 12 pass outputs
    audio = results[13]

    print(f"\n  ALBERT input shape: {albert_input.shape}")
    print(f"  ALBERT input stats: mean={albert_input.mean():.4f}, std={albert_input.std():.4f}, "
          f"min={albert_input.min():.4f}, max={albert_input.max():.4f}")

    for i, po in enumerate(pass_outputs):
        print(f"  Pass {i:2d} output: mean={po.mean():.4f}, std={po.std():.4f}, "
              f"min={po.min():.4f}, max={po.max():.4f}")

    print(f"\n  Audio shape: {audio.shape}, range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save ALBERT input as bf16 (for NPU albert_bench.exe)
    # Shape is [1, 128, 768] — squeeze batch dim to [128, 768]
    albert_in_2d = albert_input.squeeze(0)  # [128, 768]
    assert albert_in_2d.shape == (128, 768), f"Unexpected shape: {albert_in_2d.shape}"

    bf16_input = f32_to_bf16(albert_in_2d)
    bf16_path = os.path.join(outdir, "X_init.bin")
    bf16_input.tofile(bf16_path)
    print(f"\n  Saved ALBERT input (bf16): {bf16_path} ({os.path.getsize(bf16_path)} bytes)")

    # Save FP32 reference for each pass
    f32_input_path = os.path.join(outdir, "X_init_f32.bin")
    albert_in_2d.astype(np.float32).tofile(f32_input_path)
    print(f"  Saved ALBERT input (f32): {f32_input_path}")

    for i, po in enumerate(pass_outputs):
        po_2d = po.squeeze(0)  # [128, 768]
        assert po_2d.shape == (128, 768), f"Pass {i} unexpected shape: {po_2d.shape}"

        # Save FP32 reference
        f32_path = os.path.join(outdir, f"X_pass_{i}_f32.bin")
        po_2d.astype(np.float32).tofile(f32_path)

        # Save BF16 version
        bf16_po = f32_to_bf16(po_2d)
        bf16_path = os.path.join(outdir, f"X_pass_{i}_bf16_ref.bin")
        bf16_po.tofile(bf16_path)

    print(f"  Saved 12 pass outputs (f32 + bf16 reference)")

    # Save model inputs for reproducibility
    np.save(os.path.join(outdir, "tokens.npy"), tokens)
    np.save(os.path.join(outdir, "style.npy"), style)

    # Cleanup temp model
    os.remove(tmp_model)

    print(f"\nAll outputs saved to: {outdir}")
    print("\nNext steps:")
    print(f"  1. Run NPU:  albert_bench.exe ... --real-input {outdir}\\X_init.bin --dump-dir {outdir}\\npu")
    print(f"  2. Compare:  python compare_precision.py {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to kokoro-static-128-clean.onnx (or set KOKORO_MODEL env var)")
    parser.add_argument("--outdir", default=r"C:\Users\synta\npu-tts\bare-metal\kokoro\m4_real",
                        help="Output directory for extracted tensors")
    args = parser.parse_args()
    global MODEL
    if args.model:
        MODEL = args.model
    extract(args.outdir)
