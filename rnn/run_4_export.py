"""
Export trained GRU model to ONNX format with dynamic batch size.
Compatible with PyTorch 2.x and ONNX Runtime inference.
"""

import numpy as np
from pathlib import Path
import torch
import onnx
import onnxruntime as ort

from rnn.utils.const import *

MODEL_PATH = MODEL_SAVE_PATH + ".best"
ONNX_PATH = str(EXPORT_DIR / "model.onnx")


def load_model(model_path, device):
    model = MODEL(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        n_heads=N_HEADS,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    return model


def export_onnx(model, device):
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Dummy input with batch size 1 (will be dynamic)
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_SIZE, device=device)

    # Legacy exporter (dynamo=False) - dynamo has numerical issues with GRU
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"ONNX model exported to: {ONNX_PATH}")


def validate_onnx(model, device):
    """Compare PyTorch and ONNX outputs for correctness."""
    # Check model validity
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")

    # Create ONNX Runtime session
    session = ort.InferenceSession(ONNX_PATH)

    # Test with different batch sizes
    for batch_size in [1, 64, 6400]:
        test_input = torch.randn(batch_size, SEQUENCE_LENGTH, INPUT_SIZE, device=device)

        # PyTorch inference
        with torch.no_grad():
            torch_output = model(test_input).cpu().numpy()

        # ONNX inference
        ort_input = {"input": test_input.cpu().numpy()}
        ort_output = session.run(None, ort_input)[0]

        # Compare
        max_diff = np.max(np.abs(torch_output - ort_output))
        print(f"Batch {batch_size:>5d}: max absolute diff = {max_diff:.2e}")

        assert max_diff < 1e-5, f"ONNX output mismatch! Max diff: {max_diff}"

    print("All batch sizes validated successfully.")


def benchmark_onnx():
    """Benchmark ONNX Runtime inference on CPU."""
    import time

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )

    # Warmup
    dummy = np.random.randn(6400, SEQUENCE_LENGTH, INPUT_SIZE).astype(np.float32)
    for _ in range(3):
        session.run(None, {"input": dummy})

    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy})
        times.append(time.perf_counter() - t0)

    print(f"\n--- ONNX Runtime CPU Benchmark (batch=6400) ---")
    print(f"Mean: {np.mean(times) * 1000:.1f} ms")
    print(f"Min:  {np.min(times) * 1000:.1f} ms")
    print(f"Max:  {np.max(times) * 1000:.1f} ms")
    print(f"Std:  {np.std(times) * 1000:.1f} ms")


def main():
    ensure_output_dirs()

    device = torch.device("cpu")

    print("Loading model...")
    model = load_model(MODEL_PATH, device)

    print("Exporting to ONNX...")
    export_onnx(model, device)

    print("\nValidating ONNX export...")
    validate_onnx(model, device)

    print("\nBenchmarking...")
    benchmark_onnx()


if __name__ == "__main__":
    main()
