import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rnn.utils.const import INVERSE_MODEL, TARGET_MODE, WINDOW_COORD_MODE


def _iter_segments(df):
    boundary = np.zeros(len(df), dtype=bool)
    if len(df) == 0:
        return

    boundary[0] = True

    if 'experiment_name' in df.columns:
        boundary |= df['experiment_name'].ne(df['experiment_name'].shift()).to_numpy()

    if 'iteration' in df.columns:
        boundary |= df['iteration'].ne(df['iteration'].shift()).to_numpy()

    if 'step' in df.columns:
        step = pd.to_numeric(df['step'], errors='coerce')
        prev_step = step.shift()
        valid_step_pair = step.notna() & prev_step.notna()
        boundary |= (valid_step_pair & step.le(prev_step)).to_numpy()

    segment_ids = np.cumsum(boundary)
    for _, segment in df.groupby(segment_ids, sort=False):
        yield segment.copy().reset_index(drop=True)


def load_and_concat_files(file_paths):
    all_dfs = []
    print(f"Loading and concatenating {len(file_paths)} files...")
    for file_path in file_paths:
        try:
            df = pd.read_json(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Reading error in {file_path}: {e}")

    if not all_dfs:
        raise ValueError("No valid data loaded.")

    # ignore_index=True ensures the index is reset in the concatenated DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    # sort by timestamp
    full_df = full_df.sort_values(by='timestamp').reset_index(drop=True)
    print(f"Concatenated data shape: {full_df.shape}")
    return full_df


def create_windows(df, window_size):
    if len(df) < window_size:
        raise ValueError(
            "DataFrame length is less than window size. Cannot create sequences.")

    all_rel_X_windows = []
    all_rel_Y_targets = []

    for segment in _iter_segments(df):
        if len(segment) < window_size:
            continue

        if INVERSE_MODEL:
            inputs = segment[['x_actual_abs', 'y_actual_abs']].values
            outputs = segment[['x_target_abs', 'y_target_abs']].values
        else:
            inputs = segment[['x_target_abs', 'y_target_abs']].values
            outputs = segment[['x_actual_abs', 'y_actual_abs']].values

        n_samples = len(inputs)
        for i in range(n_samples - window_size + 1):
            # First point in window as reference (2,)
            # It will be subtracted to make data relative (window will start at (0,0))
            ref_X = inputs[i]

            # (window_size, 2)
            seq_X_abs = inputs[i: i + window_size]
            if WINDOW_COORD_MODE == "relative":
                seq_X = seq_X_abs - ref_X
            elif WINDOW_COORD_MODE == "delta":
                seq_X = np.zeros_like(seq_X_abs)
                seq_X[1:] = np.diff(seq_X_abs, axis=0)
            else:
                raise ValueError(f"Unsupported WINDOW_COORD_MODE: {WINDOW_COORD_MODE}")

            # Target Y
            # Last point in window
            # (2,)
            target_idx = i + window_size - 1
            target_Y_abs = outputs[target_idx]
            if WINDOW_COORD_MODE == "relative":
                target_Y = target_Y_abs - ref_X
            elif WINDOW_COORD_MODE == "delta":
                if window_size == 1:
                    target_Y = np.zeros(2, dtype=target_Y_abs.dtype)
                elif INVERSE_MODEL:
                    command_delta = target_Y_abs - outputs[target_idx - 1]
                    if TARGET_MODE == "actual_delta":
                        target_Y = command_delta
                    elif TARGET_MODE == "residual_delta":
                        desired_delta = inputs[target_idx] - inputs[target_idx - 1]
                        target_Y = command_delta - desired_delta
                    else:
                        raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
                else:
                    actual_delta = target_Y_abs - outputs[target_idx - 1]
                    if TARGET_MODE == "actual_delta":
                        target_Y = actual_delta
                    elif TARGET_MODE == "residual_delta":
                        command_delta = inputs[target_idx] - inputs[target_idx - 1]
                        target_Y = actual_delta - command_delta
                    else:
                        raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
            else:
                raise ValueError(f"Unsupported WINDOW_COORD_MODE: {WINDOW_COORD_MODE}")

            # Save the window and target
            all_rel_X_windows.append(seq_X)
            all_rel_Y_targets.append(target_Y)

    if not all_rel_X_windows:
        raise ValueError(
            "No valid data found to fit scalers. Check the input files and window size.")

    return np.array(all_rel_X_windows), np.array(all_rel_Y_targets)


def fit_scalers(X_data, Y_data):
    num_samples, window_size, num_features = X_data.shape
    assert num_features == 2, f"Expected 2 features in X_data, got {num_features}"

    reshaped_X = X_data.reshape(-1, num_features)
    all_values = np.concatenate([reshaped_X.flatten(), Y_data.flatten()], axis=0)
    all_values = all_values.reshape(-1, 1)

    scaler_global = MinMaxScaler(feature_range=(-1, 1))
    scaler_global.fit(all_values)

    print(f"Scaler fit completed on {WINDOW_COORD_MODE} data (Input features: 2).")
    return scaler_global


def scale_data(X_data, Y_data, scaler):
    num_samples, window_size, num_features = X_data.shape
    assert num_features == 2, f"Expected 2 features in X_data, got {num_features}"

    # Flatten → transform → reshape
    X_flat = X_data.reshape(-1, 1)
    X_scaled = scaler.transform(X_flat).reshape(num_samples, window_size, num_features)

    Y_flat = Y_data.reshape(-1, 1)
    Y_scaled = scaler.transform(Y_flat).reshape(num_samples, 2)

    print("Data scaling completed.")
    return X_scaled, Y_scaled
