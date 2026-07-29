import glob
from os.path import join, isdir, isfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from rnn.utils.const import SEQUENCE_LENGTH, SEED, EXPERIMENT_DIR, SCALER_PATH, DATASET_DIR, \
    DATASET_POSTFIX, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, INPUT_SIZE, REPO_ROOT, ensure_output_dirs, \
    INVERSE_MODEL, WINDOW_COORD_MODE, TARGET_MODE

if INPUT_SIZE == 2:
    from rnn.preprocessing.single import create_windows, scale_data, fit_scalers

    print(f"Using preprocessing for INPUT_SIZE = 2")
elif INPUT_SIZE == 4:
    from rnn.preprocessing.double import create_windows, scale_data, fit_scalers

    print(f"Using preprocessing for INPUT_SIZE = 4")

EXPERIMENTS = [
    "run33-complex/hysteresis_dataset_20251104_174024.jsonl",
    "run34-random678mag/hysteresis_dataset_20251114_102017.jsonl",
    "run35-random9-11-12/hysteresis_dataset_20251114_125908.jsonl",
    "run36-sawtooth-decreasing/hysteresis_dataset_20251118_170631.jsonl",
    "run37-sawtooth-complex-x/hysteresis_dataset_20251128_154751.jsonl",
    "run55-random-walk-20um/hysteresis_dataset_20260210_125219_updated.jsonl",
    "run70-data-feast-overnight-sub0/confidence_0.8_no_axis_outliers_segments",
    "run70-data-feast-overnight-sub1/confidence_0.8_no_axis_outliers_segments",
    "run70-data-feast-overnight-sub2/confidence_0.8_no_axis_outliers_segments",
    "run71-data-feast-overnight/confidence_0.7_segments",
    "run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
]


def resolve_experiment_files(root_dir, experiments=None):
    if not experiments:
        return sorted(glob.glob(join(root_dir, '*.jsonl')))

    files = []
    for experiment in experiments:
        path = join(root_dir, experiment)

        if isdir(path):
            files.extend(sorted(glob.glob(join(path, '*.jsonl'))))
        elif isfile(path) and path.endswith('.jsonl'):
            files.append(path)
        else:
            raise FileNotFoundError(f"Experiment path is not a .jsonl file or directory: {path}")

    return files


def load_dataset_file(file_path):
    path = Path(file_path)

    with path.open("r", encoding="utf-8-sig") as handle:
        first_nonspace = ""
        while True:
            char = handle.read(1)
            if not char:
                break
            if not char.isspace():
                first_nonspace = char
                break

    if not first_nonspace:
        raise ValueError("File is empty")

    if first_nonspace == "[":
        df = pd.read_json(path)
    else:
        df = pd.read_json(path, lines=True)

    if df.empty:
        raise ValueError("No rows were loaded")

    return df


def report_dataset_sanity(df, file_path):
    required_columns = [
        "timestamp",
        "x_target_abs",
        "y_target_abs",
        "x_actual_abs",
        "y_actual_abs",
    ]
    numeric_columns = required_columns[1:]

    missing_required = [column for column in required_columns if column not in df.columns]

    if missing_required:
        print(f"  Sanity warning: missing required columns: {missing_required}")

    existing_required = [column for column in required_columns if column in df.columns]
    if existing_required:
        nan_counts = df[existing_required].isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if not nan_counts.empty:
            print(f"  Sanity warning: NaN counts: {nan_counts.to_dict()}")

    for column in numeric_columns:
        if column not in df.columns:
            continue

        parsed = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(parsed.isna().sum() - df[column].isna().sum())
        if invalid_count > 0:
            print(f"  Sanity warning: non-numeric values in {column}: {invalid_count}")

    if "timestamp" in df.columns:
        parsed_timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        invalid_timestamp_count = int(parsed_timestamp.isna().sum() - df["timestamp"].isna().sum())
        if invalid_timestamp_count > 0:
            print(f"  Sanity warning: non-numeric timestamp values: {invalid_timestamp_count}")

        valid_timestamp = parsed_timestamp.dropna()
        if len(valid_timestamp) > 1 and not valid_timestamp.is_monotonic_increasing:
            print("  Sanity note: timestamp is not sorted before sorting.")


def load_and_split_per_file(all_files):
    # Kontejnery pro všechny kusy dat
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []
    test_X_chunks, test_y_chunks = [], []
    test_meta = []

    print("Zpracovávám soubory a dělím separátně...")

    gap = SEQUENCE_LENGTH  # Velikost mezery k zamezení leakage na hranicích

    for file_path in all_files:
        try:
            df = load_dataset_file(file_path)
            report_dataset_sanity(df, file_path)
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            # 1. Vytvoření oken pro CELÝ soubor
            if INPUT_SIZE == 2:
                x_w, y_w = create_windows(df, SEQUENCE_LENGTH)
            elif INPUT_SIZE == 4:
                x_w, y_w = create_windows(df, SEQUENCE_LENGTH)
            else:
                continue

            n_samples = len(x_w)
            if n_samples == 0: continue

            # 2. Slicing
            idx_train_end = int(n_samples * TRAIN_SPLIT)
            idx_val_end = int(n_samples * (TRAIN_SPLIT + VAL_SPLIT))

            # 3. Slicing with gaps to prevent data leakage
            # take the full training set
            x_tr = x_w[:idx_train_end]
            y_tr = y_w[:idx_train_end]

            if VAL_SPLIT > 0:
                # Validation start a 'gap' later after idx_train_end
                val_start = idx_train_end + gap
                x_v = x_w[val_start:idx_val_end]
                y_v = y_w[val_start:idx_val_end]
            else:
                x_v = np.empty((0, SEQUENCE_LENGTH, INPUT_SIZE))
                y_v = np.empty((0, 2))

            if TEST_SPLIT > 0:
                # Test start a 'gap' later after idx_val_end
                test_start = idx_val_end + gap
                x_te = x_w[test_start:]
                y_te = y_w[test_start:]
            else:
                test_start = n_samples
                x_te = np.empty((0, SEQUENCE_LENGTH, INPUT_SIZE))
                y_te = np.empty((0, 2))

            print(f"Processed file {file_path}: ")
            print(f"  Train samples: {len(x_tr)}")
            print(f"  Val samples:   {len(x_v)}")
            print(f"  Test samples:  {len(x_te)}")

            if len(x_tr) > 0:
                train_X_chunks.append(x_tr)
                train_y_chunks.append(y_tr)

            if len(x_v) > 0:
                val_X_chunks.append(x_v)
                val_y_chunks.append(y_v)

            if len(x_te) > 0:
                test_X_chunks.append(x_te)
                test_y_chunks.append(y_te)
                for window_idx in range(test_start, n_samples):
                    test_meta.append({
                        "source_file": file_path,
                        "window_idx_in_file": window_idx,
                        "raw_start_idx": window_idx,
                        "raw_end_idx": window_idx + SEQUENCE_LENGTH - 1,
                    })

        except (ValueError, OSError) as e:
            print(f"Skipping file {file_path}: {e}")

    # 4. Concat all chunks together
    def concat_chunks(chunks):
        if not chunks:
            return np.empty((0, SEQUENCE_LENGTH, INPUT_SIZE))  # Ošetření prázdného
        return np.concatenate(chunks, axis=0)

    def concat_y(chunks):
        if not chunks:
            return np.empty((0, 2))
        return np.concatenate(chunks, axis=0)

    X_train = concat_chunks(train_X_chunks)
    y_train = concat_y(train_y_chunks)

    X_val = concat_chunks(val_X_chunks)
    y_val = concat_y(val_y_chunks)

    X_test = concat_chunks(test_X_chunks)
    y_test = concat_y(test_y_chunks)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_meta


def plot_windows(X, y, num_windows=5):
    import matplotlib.pyplot as plt

    for i in range(min(num_windows, len(X))):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(X[i])), X[i, :, 0], 'r', label='X_tx', linestyle=':')
        plt.plot(range(len(X[i])), X[i, :, 1], 'c', label='X_ty', linestyle=':')
        plt.plot(len(X[i]), y[i, 0], 'ro', label='y_actual X')
        plt.plot(len(X[i]), y[i, 1], 'go', label='y_actual Y')

        if INPUT_SIZE == 4:
            plt.plot(range(len(X[i])), X[i, :, 2], 'ro', label='X_ax', linestyle=':')
            plt.plot(range(len(X[i])), X[i, :, 3], 'go', label='X_ay', linestyle=':')
        plt.title(f'Window {i + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


def inverse_transform_values(data, scaler):
    original_shape = data.shape
    flat = data.reshape(-1, 1)
    inverse_flat = scaler.inverse_transform(flat)
    return inverse_flat.reshape(original_shape)


def scale_data_or_empty(X_data, y_data, scaler):
    if len(X_data) == 0:
        return X_data, y_data
    return scale_data(X_data, y_data, scaler)


def print_plotted_window_debug(X_scaled, y_scaled, X_unscaled, y_unscaled, test_meta, plot_index, scaler):
    if plot_index >= len(X_scaled):
        print(f"Cannot debug Window {plot_index + 1}: only {len(X_scaled)} plotted test windows available.")
        return

    meta = test_meta[plot_index] if plot_index < len(test_meta) else None
    X_inv = inverse_transform_values(X_scaled[plot_index], scaler)
    y_inv = inverse_transform_values(y_scaled[plot_index], scaler)

    print("\n" + "=" * 80)
    print(f"DEBUG PLOTTED WINDOW {plot_index + 1}")
    print("=" * 80)
    print(f"WINDOW_COORD_MODE/TARGET_MODE are defined in const.py.")
    if meta:
        print(f"Source file: {meta['source_file']}")
        print(f"Window index in file: {meta['window_idx_in_file']}")
        print(f"Raw rows: {meta['raw_start_idx']}..{meta['raw_end_idx']}")

        try:
            df = load_dataset_file(meta["source_file"])
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            raw_start = meta["raw_start_idx"]
            raw_end = meta["raw_end_idx"]
            cols = [
                "experiment_name",
                "iteration",
                "step",
                "timestamp",
                "x_target_abs",
                "y_target_abs",
                "x_actual_abs",
                "y_actual_abs",
            ]
            cols = [col for col in cols if col in df.columns]
            print("\nRaw trajectory rows:")
            print(df.loc[raw_start:raw_end, cols].to_string(index=True))

            targets = df[["x_target_abs", "y_target_abs"]].to_numpy(dtype=float)
            actuals = df[["x_actual_abs", "y_actual_abs"]].to_numpy(dtype=float)
            target_idx = raw_end

            if target_idx > 0:
                command_delta = targets[target_idx] - targets[target_idx - 1]
                actual_delta = actuals[target_idx] - actuals[target_idx - 1]
                print("\nLast-step label computation:")
                print(f"  previous raw row: {target_idx - 1}")
                print(f"  target raw row:   {target_idx}")
                print(f"  command_delta = ({command_delta[0]:.1f}, {command_delta[1]:.1f}) nm")
                print(f"  actual_delta  = ({actual_delta[0]:.1f}, {actual_delta[1]:.1f}) nm")
                if WINDOW_COORD_MODE == "delta" and TARGET_MODE == "residual_delta":
                    if INVERSE_MODEL:
                        residual_delta = command_delta - actual_delta
                        print(
                            "  residual_delta = command_delta - desired_delta "
                            f"= ({residual_delta[0]:.1f}, {residual_delta[1]:.1f}) nm"
                        )
                    else:
                        residual_delta = actual_delta - command_delta
                        print(
                            "  residual_delta = actual_delta - command_delta "
                            f"= ({residual_delta[0]:.1f}, {residual_delta[1]:.1f}) nm"
                        )
        except (ValueError, OSError, KeyError) as exc:
            print(f"Could not print raw trajectory for debug window: {exc}")

    print("\nUnscaled X window values reconstructed from scaler (nm):")
    if INPUT_SIZE == 4:
        if INVERSE_MODEL:
            print("t  desired_dx desired_dy  prev_command_dx prev_command_dy")
        else:
            print("t  command_dx command_dy  prev_actual_dx prev_actual_dy")
        for t, row in enumerate(X_inv):
            print(f"{t:2d} {row[0]:10.1f} {row[1]:10.1f} {row[2]:15.1f} {row[3]:15.1f}")
    else:
        print("t  dx         dy")
        for t, row in enumerate(X_inv):
            print(f"{t:2d} {row[0]:10.1f} {row[1]:10.1f}")

    print("\nScaled X window values used by plot:")
    if INPUT_SIZE == 4:
        if INVERSE_MODEL:
            print("t  X_des_x   X_des_y   X_prev_cmd_x X_prev_cmd_y")
        else:
            print("t  X_cmd_x   X_cmd_y   X_prev_act_x X_prev_act_y")
        for t, row in enumerate(X_scaled[plot_index]):
            print(f"{t:2d} {row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f} {row[3]:9.6f}")
    else:
        print("t  X_x       X_y")
        for t, row in enumerate(X_scaled[plot_index]):
            print(f"{t:2d} {row[0]:9.6f} {row[1]:9.6f}")

    print("\nLabel:")
    print(f"  unscaled y from scaler: ({y_inv[0]:.1f}, {y_inv[1]:.1f}) nm")
    print(f"  unscaled y before scaler: ({y_unscaled[plot_index, 0]:.1f}, {y_unscaled[plot_index, 1]:.1f}) nm")
    print(f"  scaled y used by plot: ({y_scaled[plot_index, 0]:.6f}, {y_scaled[plot_index, 1]:.6f})")
    print("=" * 80 + "\n")


def main():
    ensure_output_dirs()

    # determinism
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_files = resolve_experiment_files(str(REPO_ROOT / EXPERIMENT_DIR), EXPERIMENTS)
    print(f"Found {len(all_files)} dataset files in {EXPERIMENT_DIR}.")
    for file_path in all_files:
        print(f"  {file_path}")

    (X_train_u, y_train_u), (X_val_u, y_val_u), (X_test_u, y_test_u), test_meta = load_and_split_per_file(all_files)

    print(f"Train samples: {len(X_train_u)}")
    print(f"Val samples:   {len(X_val_u)}")
    print(f"Test samples:  {len(X_test_u)}")

    if len(X_train_u) == 0:
        raise ValueError("No training samples were generated from the selected dataset files.")

    # shuffle before splitting
    perm = np.random.permutation(len(X_train_u))
    X_train_u = X_train_u[perm]
    y_train_u = y_train_u[perm]

    # fitting the scalers on unscaled training data
    print("Generating and scaling dataset...")
    scaler = fit_scalers(X_train_u, y_train_u)

    # save scalers
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scalers saved.")

    # Aplikace scalerů
    X_train, y_train = scale_data(X_train_u, y_train_u, scaler)
    X_val, y_val = scale_data_or_empty(X_val_u, y_val_u, scaler)
    X_test, y_test = scale_data_or_empty(X_test_u, y_test_u, scaler)

    # 5. Tvorba TensorDataset
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    print("Datasets created.")

    # 6. Uložení
    torch.save(train_dataset, f"{DATASET_DIR}train{DATASET_POSTFIX}")
    torch.save(val_dataset, f"{DATASET_DIR}val{DATASET_POSTFIX}")
    torch.save(test_dataset, f"{DATASET_DIR}test{DATASET_POSTFIX}")
    print(f"Datasets saved to {DATASET_DIR} directory.")

    # shuffle before plotting
    perm_test = np.random.permutation(len(X_test))
    X_test = X_test[perm_test]
    y_test = y_test[perm_test]
    X_test_u = X_test_u[perm_test]
    y_test_u = y_test_u[perm_test]
    test_meta = [test_meta[i] for i in perm_test]

    print_plotted_window_debug(
        X_test,
        y_test,
        X_test_u,
        y_test_u,
        test_meta,
        plot_index=5,
        scaler=scaler,
    )

    plot_windows(X_test, y_test, num_windows=10)


if __name__ == "__main__":
    main()
