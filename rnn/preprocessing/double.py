import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


def _fill_previous_from_current(df, previous_column, current_column):
    df[previous_column] = df[current_column].shift(1)
    df[previous_column] = df[previous_column].fillna(df[current_column])


def create_windows(df, window_size, min_confidence):
    if len(df) < window_size:
        raise ValueError(
            "DataFrame length is less than window size. Cannot create sequences.")

    all_rel_X_windows = []
    all_rel_Y_targets = []

    n_low_conf = 0
    try:
        for segment in _iter_segments(df):
            if len(segment) < window_size:
                continue

            # 1. Prepare raw data (N, 2). Previous values are segment-local;
            # the first row uses itself, producing zero previous-delta.
            if INVERSE_MODEL:
                _fill_previous_from_current(segment, 'prev_x_target', 'x_target_abs')
                _fill_previous_from_current(segment, 'prev_y_target', 'y_target_abs')

                actuals = segment[['x_actual_abs', 'y_actual_abs']].values  # "desired positions"
                targets = segment[['x_target_abs', 'y_target_abs']].values  # "commands needed"
                prev_targets = segment[['prev_x_target', 'prev_y_target']].values

                # Combined: [actual_x, actual_y, prev_target_x, prev_target_y]
                combined_data = np.hstack([actuals, prev_targets])
            else:
                _fill_previous_from_current(segment, 'prev_x_actual', 'x_actual_abs')
                _fill_previous_from_current(segment, 'prev_y_actual', 'y_actual_abs')
                true_actuals = segment[['x_actual_abs', 'y_actual_abs']].values

                targets = segment[['x_target_abs', 'y_target_abs']].values
                prev_actuals = segment[['prev_x_actual', 'prev_y_actual']].values

                combined_data = np.hstack([targets, prev_actuals])

            # confidences as numpy array for fast indexing
            confidences = segment['confidence'].astype(float).values

            n_samples = len(combined_data)
            for i in range(n_samples - window_size + 1):

                window_conf = confidences[i: i + window_size]
                if np.any(window_conf < min_confidence):
                    n_low_conf += 1
                    continue

                if INVERSE_MODEL:
                    ref_X = actuals[i]  # reference = desired position
                    target_Y_abs = targets[i + window_size - 1]  # predict = command needed
                else:
                    ref_X = targets[i]  # reference = command
                    target_Y_abs = true_actuals[i + window_size - 1]  # predict = actual position

                seq_abs = combined_data[i: i + window_size].copy()
                if WINDOW_COORD_MODE == "relative":
                    ref_full = np.hstack([ref_X, ref_X])
                    seq_X = seq_abs - ref_full
                    target_Y = target_Y_abs - ref_X
                elif WINDOW_COORD_MODE == "delta":
                    seq_X = np.zeros_like(seq_abs)
                    seq_X[1:] = np.diff(seq_abs, axis=0)
                    if window_size == 1:
                        target_Y = np.zeros(2, dtype=target_Y_abs.dtype)
                    elif INVERSE_MODEL:
                        target_Y = target_Y_abs - targets[i + window_size - 2]
                    else:
                        actual_delta = target_Y_abs - true_actuals[i + window_size - 2]
                        if TARGET_MODE == "actual_delta":
                            target_Y = actual_delta
                        elif TARGET_MODE == "residual_delta":
                            command_delta = targets[i + window_size - 1] - targets[i + window_size - 2]
                            target_Y = actual_delta - command_delta
                        else:
                            raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
                else:
                    raise ValueError(f"Unsupported WINDOW_COORD_MODE: {WINDOW_COORD_MODE}")

                all_rel_X_windows.append(seq_X)
                all_rel_Y_targets.append(target_Y)

    except Exception as e:
        print(f"Reading error in DataFrame processing: {e}")

    if not all_rel_X_windows:
        raise ValueError("No valid data found after filtering out low-confidence windows.")

    print(
        f"Created {len(all_rel_X_windows)} windows of size {window_size} after filtering out {n_low_conf} low-confidence windows.")

    return np.array(all_rel_X_windows), np.array(all_rel_Y_targets)


def fit_scalers(X_data, Y_data):
    num_samples, window_size, num_features = X_data.shape

    # Now we expect 4 features: [Tx, Ty, Ax, Ay]
    assert num_features == 4, f"Expected 4 features in X_data, got {num_features}"

    reshaped_X = np.array(X_data).reshape(-1, num_features)

    all_values = np.concatenate([reshaped_X.flatten(), Y_data.flatten()], axis=0)
    all_values = all_values.reshape(-1, 1)

    scaler_global = MinMaxScaler(feature_range=(-1, 1))
    scaler_global.fit(all_values)

    print(f"Scalers fit completed on {WINDOW_COORD_MODE} data (Input features: 4).")

    return scaler_global


def scale_data(X_data, Y_data, scaler):
    num_samples, window_size, num_features = X_data.shape

    assert num_features == 4, f"Expected 4 features in X_data, got {num_features}"

    # Scale X_data
    X_flat = X_data.reshape(-1, 1)
    X_scaled_flat = scaler.transform(X_flat)
    scaled_X_data = X_scaled_flat.reshape(num_samples, window_size, num_features)

    # --- Scale Y ---
    Y_flat = Y_data.reshape(-1, 1)
    Y_scaled_flat = scaler.transform(Y_flat)
    scaled_Y_data = Y_scaled_flat.reshape(num_samples, 2)

    print("Data scaling completed.")

    return scaled_X_data, scaled_Y_data
