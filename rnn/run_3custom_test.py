import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from rnn.utils.const import *

CUSTOM_TEST_MANIFEST_PATH = f"{DATASET_DIR}manifest.json"
CUSTOM_TEST_RESULTS_PATH = f"{DATASET_DIR}custom_test_results.csv"


def inverse_transform_helper(data, scaler):
    original_shape = data.shape
    data_flat = data.reshape(-1, 1)
    data_inv_flat = scaler.inverse_transform(data_flat)
    return data_inv_flat.reshape(original_shape)


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
        new_state_dict = {
            key.replace("_orig_mod.", ""): value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(new_state_dict)

    model.eval()
    return model


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, list):
        raise ValueError(f"Expected list manifest at {manifest_path}")

    return manifest


def evaluate_dataset(model, scaler, dataset_path, device):
    dataset = torch.load(dataset_path, weights_only=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(sequences.cpu().numpy())

    if not all_preds:
        raise ValueError("Dataset contains no batches")

    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true_scaled = np.concatenate(all_labels, axis=0)
    X_input_scaled = np.concatenate(all_inputs, axis=0)

    y_pred_nm = inverse_transform_helper(y_pred_scaled, scaler)
    y_true_nm = inverse_transform_helper(y_true_scaled, scaler)
    X_input_nm = inverse_transform_helper(X_input_scaled, scaler)

    y_pred_naive_nm = X_input_nm[:, -1, 0:2]

    mse = mean_squared_error(y_true_nm, y_pred_nm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_nm, y_pred_nm)
    r2 = r2_score(y_true_nm, y_pred_nm)
    nrmse = rmse / (y_true_nm.max() - y_true_nm.min())
    rmse_x = np.sqrt(mean_squared_error(y_true_nm[:, 0], y_pred_nm[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_true_nm[:, 1], y_pred_nm[:, 1]))

    baseline_rmse = np.sqrt(mean_squared_error(y_true_nm, y_pred_naive_nm))
    baseline_mae = mean_absolute_error(y_true_nm, y_pred_naive_nm)
    baseline_r2 = r2_score(y_true_nm, y_pred_naive_nm)

    return {
        "samples": len(dataset),
        "rmse_nm": rmse,
        "mae_nm": mae,
        "r2": r2,
        "nrmse": nrmse,
        "rmse_x_nm": rmse_x,
        "rmse_y_nm": rmse_y,
        "baseline_rmse_nm": baseline_rmse,
        "baseline_mae_nm": baseline_mae,
        "baseline_r2": baseline_r2,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = f"{MODEL_SAVE_PATH}.best"
    required_paths = [CUSTOM_TEST_MANIFEST_PATH, model_path, SCALER_PATH]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("Error: Missing files:")
        for path in missing_paths:
            print(f"  {path}")
        return

    manifest = load_manifest(CUSTOM_TEST_MANIFEST_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(model_path, device)

    print(f"Loaded model from {model_path}")
    print(f"Loaded scaler from {SCALER_PATH}")
    print(f"Loaded {len(manifest)} custom test dataset entries.")

    rows = []
    for entry in manifest:
        source_file = entry.get("source_file", "")
        dataset_path = entry.get("dataset_path", "")

        if not dataset_path or not os.path.exists(dataset_path):
            print(f"Skipping missing dataset for {source_file}: {dataset_path}")
            continue

        try:
            metrics = evaluate_dataset(model, scaler, dataset_path, device)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"Skipping failed dataset {dataset_path}: {exc}")
            continue

        row = {
            "source_file": source_file,
            "dataset_path": dataset_path,
            **metrics,
        }
        rows.append(row)

        print("\n--- Custom Test Dataset ---")
        print(f"Source: {source_file}")
        print(f"Dataset: {dataset_path}")
        print(f"Samples: {metrics['samples']}")
        print(f"Model RMSE: {metrics['rmse_nm']:.4f} nm")
        print(f"Model MAE:  {metrics['mae_nm']:.4f} nm")
        print(f"Model R2:   {metrics['r2']:.4f}")
        print(f"Baseline RMSE: {metrics['baseline_rmse_nm']:.4f} nm")
        print(f"Baseline MAE:  {metrics['baseline_mae_nm']:.4f} nm")
        print(f"Baseline R2:   {metrics['baseline_r2']:.4f}")

    if not rows:
        print("No custom test datasets were evaluated.")
        return

    Path(CUSTOM_TEST_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "dataset_path",
        "samples",
        "rmse_nm",
        "mae_nm",
        "r2",
        "nrmse",
        "rmse_x_nm",
        "rmse_y_nm",
        "baseline_rmse_nm",
        "baseline_mae_nm",
        "baseline_r2",
    ]
    with open(CUSTOM_TEST_RESULTS_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved custom test results to {CUSTOM_TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
