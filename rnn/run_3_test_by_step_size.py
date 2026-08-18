"""Evaluate the standard test split in ground-truth command-size buckets.

The standard ``test.pt`` stores only tensors, so it cannot be grouped by the
source experiment.  This script groups the same predictions used by
``run_3_test.py`` by the magnitude of the true motor command delta.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Support both ``python -m rnn.run_3_test_by_step_size`` and direct execution.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rnn.run_3_test import inverse_transform_helper, to_actual_delta_metrics_space
from rnn.run_3custom_test import load_model
from rnn.utils.const import (
    BATCH_SIZE,
    DATASET_DIR_PATH,
    DATASET_POSTFIX,
    INVERSE_MODEL,
    MODEL_SAVE_PATH,
    SCALER_PATH,
    SEED,
    WINDOW_COORD_MODE,
)


STEP_BINS_NM = (
    # Scaler round-trips represent a true zero as a few millionths of a nm.
    (0.0, 0.5, "0 nm (no movement)"),
    (0.5, 100.0, ">0-100 nm"),
    (100.0, 300.0, "100-300 nm"),
    (300.0, 1_000.0, "300-1000 nm"),
    (1_000.0, 3_000.0, "1-3 um"),
    (3_000.0, np.inf, ">=3 um"),
)

EVALUATION_BATCH_SIZE = max(BATCH_SIZE, 512)

TEST_DATASET_PATH = DATASET_DIR_PATH / f"test{DATASET_POSTFIX}"
RESULTS_PATH = DATASET_DIR_PATH / "test_results_by_step_size.csv"


def run_inference(model, dataset, device):
    loader = DataLoader(dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False)
    predictions = []
    labels = []
    inputs = []

    with torch.no_grad():
        for sequences, batch_labels in loader:
            sequences = sequences.to(device)
            predictions.append(model(sequences).cpu().numpy())
            labels.append(batch_labels.numpy())
            inputs.append(sequences.cpu().numpy())

    if not predictions:
        raise ValueError(f"Test dataset is empty: {TEST_DATASET_PATH}")

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(inputs, axis=0),
    )


def calculate_bucket_metrics(
    step_size_nm,
    y_pred_nm,
    y_true_nm,
    y_baseline_nm,
):
    model_error = y_pred_nm - y_true_nm
    baseline_error = y_baseline_nm - y_true_nm
    model_error_distance = np.linalg.norm(model_error, axis=1)
    baseline_error_distance = np.linalg.norm(baseline_error, axis=1)

    model_rmse = float(np.sqrt(np.mean(np.square(model_error))))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    if baseline_rmse > 0:
        rmse_improvement_pct = 100.0 * (baseline_rmse - model_rmse) / baseline_rmse
    else:
        rmse_improvement_pct = np.nan

    return {
        "samples": len(step_size_nm),
        "share_pct": 0.0,
        "step_mean_nm": float(np.mean(step_size_nm)),
        "step_median_nm": float(np.median(step_size_nm)),
        "model_mae_nm": float(np.mean(np.abs(model_error))),
        "model_rmse_nm": model_rmse,
        "model_rmse_x_nm": float(np.sqrt(np.mean(np.square(model_error[:, 0])))),
        "model_rmse_y_nm": float(np.sqrt(np.mean(np.square(model_error[:, 1])))),
        "model_error_distance_p50_nm": float(np.percentile(model_error_distance, 50)),
        "model_error_distance_p90_nm": float(np.percentile(model_error_distance, 90)),
        "baseline_mae_nm": float(np.mean(np.abs(baseline_error))),
        "baseline_rmse_nm": baseline_rmse,
        "baseline_error_distance_p50_nm": float(
            np.percentile(baseline_error_distance, 50)
        ),
        "rmse_improvement_pct": rmse_improvement_pct,
    }


def build_step_size_table(
    step_size_nm,
    y_pred_nm,
    y_true_nm,
    y_baseline_nm,
):
    rows = []
    total_samples = len(step_size_nm)

    for lower_nm, upper_nm, label in STEP_BINS_NM:
        mask = (step_size_nm >= lower_nm) & (step_size_nm < upper_nm)
        if not np.any(mask):
            continue

        row = {
            "step_size": label,
            **calculate_bucket_metrics(
                step_size_nm[mask],
                y_pred_nm[mask],
                y_true_nm[mask],
                y_baseline_nm[mask],
            ),
        }
        row["share_pct"] = 100.0 * row["samples"] / total_samples
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if WINDOW_COORD_MODE != "delta":
        raise ValueError(
            "Step-size grouping currently requires WINDOW_COORD_MODE='delta'"
        )

    model_path = Path(f"{MODEL_SAVE_PATH}.best")
    required_paths = (TEST_DATASET_PATH, model_path, Path(SCALER_PATH))
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"  {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required files:\n{missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(SCALER_PATH)
    model = load_model(model_path, device)
    test_dataset = torch.load(TEST_DATASET_PATH, weights_only=False)

    y_pred_scaled, y_true_scaled, x_input_scaled = run_inference(
        model,
        test_dataset,
        device,
    )
    y_pred_nm = inverse_transform_helper(y_pred_scaled, scaler)
    y_true_nm = inverse_transform_helper(y_true_scaled, scaler)
    x_input_nm = inverse_transform_helper(x_input_scaled, scaler)

    y_pred_eval_nm, y_true_eval_nm, y_baseline_nm, metric_target = (
        to_actual_delta_metrics_space(y_pred_nm, y_true_nm, x_input_nm)
    )

    # Bucket by the ground-truth motor command, never by actual/observed motion.
    # For the inverse model to_actual_delta_metrics_space() reconstructs the
    # true command delta as input_delta + true residual.  For the forward model
    # the command delta is already in the first two input features.
    if INVERSE_MODEL:
        command_step_delta_nm = y_true_eval_nm
    else:
        command_step_delta_nm = x_input_nm[:, -1, 0:2]
    # Raw command coordinates are integer nanometers. Undo tiny float32/scaler
    # round-trip errors so exact boundaries (100, 300, 1000 nm, ...) do not
    # fall into the lower bucket as 99.99998, 299.99997, etc.
    command_step_delta_nm = np.rint(command_step_delta_nm)
    step_size_nm = np.linalg.norm(command_step_delta_nm, axis=1)

    table = build_step_size_table(
        step_size_nm,
        y_pred_eval_nm,
        y_true_eval_nm,
        y_baseline_nm,
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False, float_format="%.4f")

    print(f"Device: {device}")
    print(f"Test samples: {len(test_dataset)}")
    print("Step size: Euclidean norm of ground-truth command delta")
    print(f"Metric target: {metric_target}")
    print("\n--- Test results by command step size ---")
    print(table.round(2).to_string(index=False))
    print(f"\nSaved CSV to {RESULTS_PATH}")
    print(
        "Experiment tables are unavailable because the standard test.pt "
        "does not store per-sample source metadata."
    )


if __name__ == "__main__":
    main()
