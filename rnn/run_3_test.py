import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from rnn.utils.const import *


def inverse_transform_helper(data, scaler):
    """
    Pomocná funkce pro inverzní transformaci s globálním scalerem.
    Scaler očekává vstup (N, 1).
    Data mohou být (N, 2) nebo (N, Window, 4).
    """
    original_shape = data.shape

    # 1. Zploštit na (Total_Samples, 1)
    data_flat = data.reshape(-1, 1)

    # 2. Inverzní transformace
    data_inv_flat = scaler.inverse_transform(data_flat)

    # 3. Vrátit do původního tvaru
    return data_inv_flat.reshape(original_shape)


def evaluate_naive_guess(X_data_nm, y_true_nm):
    # X_data_nm shape: (N_samples, Window_Size, Features)
    # Features INPUT_SIZE=4: [Target_X, Target_Y, Prev_Actual_X, Prev_Actual_Y]

    if INVERSE_MODEL:
        print("\n--- 'No Compensation' Baseline (send desired position as command directly) ---")
    else:
        print("\n--- 'Command' Baseline (assume system follows command perfectly) ---")

    # Last time step's Target as prediction
    # Indices 0 a 1 are Target X a Target Y
    y_pred_naive_nm = X_data_nm[:, -1, 0:2]

    print("That is, if we would assume the system perfectly follows the command input.")

    rmse_naive = np.sqrt(mean_squared_error(y_true_nm, y_pred_naive_nm))
    nrmse = rmse_naive / (y_true_nm.max() - y_true_nm.min())
    mae_naive = mean_absolute_error(y_true_nm, y_pred_naive_nm)
    r2_naive = r2_score(y_true_nm, y_pred_naive_nm)

    print(f"Overall R²:     {r2_naive:.4f}")
    print(f"Overall RMSE:    {rmse_naive:.4f} nm")
    print(f"Overall MAE:     {mae_naive:.4f} nm")
    print(f"Overall NRMSE:   {nrmse:.4%}")

    # Save plot
    plt.figure(figsize=(20, 5))
    plt.suptitle(f"Baseline (Command = Output)", fontsize=14)
    # Plot only a slice to make it visible
    limit = 200
    if INVERSE_MODEL:
        plt.plot(y_true_nm[:limit, 0], label='True Command (X)', color='blue', alpha=0.8)
        plt.plot(y_pred_naive_nm[:limit, 0], label='Desired Pos (no compensation)', color='green', alpha=0.8)
    else:
        plt.plot(y_true_nm[:limit, 0], label='True Position (X)', color='blue', alpha=0.8)
        plt.plot(y_pred_naive_nm[:limit, 0], label='Command (Target X)', color='green', alpha=0.8)
    plt.title('Baseline: Command vs Reality (First 200 samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("temp/command_baseline.png")
    print("Baseline plot saved.")


def evaluate():
    # 1. Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dataset_path = f"{DATASET_DIR}test{DATASET_POSTFIX}"
    model_path = f"{MODEL_SAVE_PATH}.best"

    if not all(os.path.exists(p) for p in [test_dataset_path, model_path, SCALER_PATH, SCALER_PATH]):
        print("Error: Missing files.")
        return

    # 2. Load Scalers
    # Caution: they just may be the same
    scaler = joblib.load(SCALER_PATH)
    print("Scalers loaded.")

    # 3. Load Model
    model = MODEL(input_size=INPUT_SIZE,
                  hidden_size=HIDDEN_SIZE,
                  output_size=OUTPUT_SIZE,
                  num_layers=NUM_LAYERS,
                  dropout=DROPOUT,
                  bidirectional=BIDIRECTIONAL,
                  n_heads=N_HEADS).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # If model was trained using torch.compile, keys have prefix "_orig_mod."
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    model.eval()
    print(f"Model loaded from '{model_path}'.")

    # 4. Load Data
    test_dataset = torch.load(test_dataset_path, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test dataset loaded ({len(test_dataset)} samples).")

    # 5. Inference Loop
    all_preds = []
    all_labels = []
    all_inputs = []  # Potřebujeme i vstupy pro Baseline

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(sequences.cpu().numpy())

    # Concatenate batches
    y_pred_scaled = np.concatenate(all_preds, axis=0)  # (N, 2)
    y_true_scaled = np.concatenate(all_labels, axis=0)  # (N, 2)
    X_input_scaled = np.concatenate(all_inputs, axis=0)  # (N, Window, 4)

    # 6. Inverse Transform
    print("Inverse transforming data...")
    y_pred_nm = inverse_transform_helper(y_pred_scaled, scaler)
    y_true_nm = inverse_transform_helper(y_true_scaled, scaler)
    X_input_nm = inverse_transform_helper(X_input_scaled, scaler)

    # 7. Evaluate Baseline
    evaluate_naive_guess(X_input_nm, y_true_nm)

    # 8. Model Metrics
    print("\n--- Model Metrics (Nanometers) ---")
    rmse = np.sqrt(mean_squared_error(y_true_nm, y_pred_nm))
    mae = mean_absolute_error(y_true_nm, y_pred_nm)
    r2 = r2_score(y_true_nm, y_pred_nm)
    nrmse = rmse / (y_true_nm.max() - y_true_nm.min())

    print(f"Overall R²:     {r2:.4f}")
    print(f"Overall RMSE:    {rmse:.4f} nm")
    print(f"Overall MAE:     {mae:.4f} nm")
    print(f"Overall NRMSE:   {nrmse:.4%}")

    rmse_x = np.sqrt(mean_squared_error(y_true_nm[:, 0], y_pred_nm[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_true_nm[:, 1], y_pred_nm[:, 1]))

    print(f"X RMSE: {rmse_x:.4f} nm")
    print(f"Y RMSE: {rmse_y:.4f} nm")

    # 9. Plotting
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Error Histogram
    errors_nm = y_true_nm - y_pred_nm
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.hist(errors_nm[:, 0], bins=50, alpha=0.7, label='X Error')
    plt.hist(errors_nm[:, 1], bins=50, alpha=0.7, label='Y Error')
    plt.title('Error Distribution (nm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter
    plt.subplot(3, 2, 2)
    plt.scatter(y_true_nm[:, 0], y_pred_nm[:, 0], s=7, alpha=0.5, label='X')
    plt.scatter(y_true_nm[:, 1], y_pred_nm[:, 1], s=7, alpha=0.5, label='Y')
    plt.plot([y_true_nm.min(), y_true_nm.max()], [y_true_nm.min(), y_true_nm.max()], 'r--')
    plt.title('True vs Predicted')
    plt.xlabel('True [nm]')
    plt.ylabel('Pred [nm]')

    # Time Series Detail
    plt.subplot(3, 1, 2)
    slice_idx = slice(0, 300)  # Zobrazit prvních 300 bodů
    # plt.plot(y_true_nm[slice_idx, 0], label='True X', color='black', linewidth=2)
    # plt.plot(y_pred_nm[slice_idx, 0], label='Pred X', color='red', linestyle='--')
    plt.plot(np.abs(y_true_nm[slice_idx, 0] - y_pred_nm[slice_idx, 0]), label='Abs Error X', color='blue',
             linestyle=':')
    # Můžeme přidat i Target pro kontext
    # plt.plot(X_input_nm[slice_idx, -1, 0], label='Command X', color='green', alpha=0.3)
    plt.plot(np.abs(y_true_nm[slice_idx, 0] - X_input_nm[slice_idx, -1, 0]), label='Abs Command Error X',
             color='orange', linestyle='-.')
    plt.title('Detail: First 300 samples (X Axis)')
    plt.legend()
    plt.ylim(0, 4000)
    plt.grid(True, alpha=0.3)

    # Time Series Detail Y
    plt.subplot(3, 1, 3)
    # plt.plot(y_true_nm[slice_idx, 1], label='True Y', color='black', linewidth=2)
    # plt.plot(y_pred_nm[slice_idx, 1], label='Pred Y', color='red', linestyle='--')
    plt.plot(np.abs(y_true_nm[slice_idx, 1] - y_pred_nm[slice_idx, 1]), label='Abs Error Y', color='blue',
             linestyle=':')
    plt.plot(np.abs(y_true_nm[slice_idx, 1] - X_input_nm[slice_idx, -1, 1]), label='Abs Command Error Y',
             color='orange', linestyle='-.')
    # plt.plot(X_input_nm[slice_idx, -1, 1], label='Command Y', color='green', alpha=0.3)
    plt.title('Detail: First 300 samples (Y Axis)')
    plt.ylim(0, 4000)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("temp/evaluation_metrics.png")
    print("\nPlots saved to 'temp/evaluation_metrics.png'")


if __name__ == "__main__":
    evaluate()
