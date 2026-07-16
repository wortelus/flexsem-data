import json
import re
from os import makedirs
from os.path import exists, join
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset

from rnn.run_1_generate_dataset import load_dataset_file, report_dataset_sanity, resolve_experiment_files
from rnn.utils.const import SEQUENCE_LENGTH, SEED, EXPERIMENT_DIR, SCALER_PATH, DATASET_DIR, \
    DATASET_POSTFIX, TRAIN_SPLIT, INPUT_SIZE, MIN_CONFIDENCE

if INPUT_SIZE == 2:
    from rnn.preprocessing.single import create_windows, scale_data, fit_scalers

    print("Using preprocessing for INPUT_SIZE = 2")
elif INPUT_SIZE == 4:
    from rnn.preprocessing.double import create_windows, scale_data, fit_scalers

    print("Using preprocessing for INPUT_SIZE = 4")
else:
    raise ValueError(f"Unsupported INPUT_SIZE: {INPUT_SIZE}")

# Train/val sources are split per file into train and validation only.
TRAIN_VAL_EXPERIMENTS = [
    "run33-complex/hysteresis_dataset_20251104_174024.jsonl",
    "run34-random678mag/hysteresis_dataset_20251114_102017.jsonl",
    "run35-random9-11-12/hysteresis_dataset_20251114_125908.jsonl",
    "run36-sawtooth-decreasing/hysteresis_dataset_20251118_170631.jsonl",
    "run37-sawtooth-complex-x/hysteresis_dataset_20251128_154751.jsonl",
    "run55-random-walk-20um/hysteresis_dataset_20260210_125219_updated.jsonl",
]

# Test sources are not mixed together. Every resolved JSONL file is saved as
# its own test TensorDataset under TEST_DATASET_DIR.
TEST_EXPERIMENTS = [
    "run70-data-feast-overnight-sub0/confidence_0.8_no_axis_outliers_segments",
    "run70-data-feast-overnight-sub1/confidence_0.8_no_axis_outliers_segments",
    "run70-data-feast-overnight-sub2/confidence_0.8_no_axis_outliers_segments",
    "run71-data-feast-overnight/confidence_0.7_segments",
    "run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
]

TEST_DATASET_DIR = join(DATASET_DIR, "test_custom")
TEST_MANIFEST_PATH = join(DATASET_DIR, "manifest.json")


def create_windows_for_file(file_path):
    df = load_dataset_file(file_path)
    report_dataset_sanity(df, file_path)
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    if INPUT_SIZE == 2:
        return create_windows(df, SEQUENCE_LENGTH)
    if INPUT_SIZE == 4:
        return create_windows(df, SEQUENCE_LENGTH, min_confidence=MIN_CONFIDENCE)

    raise ValueError(f"Unsupported INPUT_SIZE: {INPUT_SIZE}")


def empty_x():
    return np.empty((0, SEQUENCE_LENGTH, INPUT_SIZE))


def empty_y():
    return np.empty((0, 2))


def concat_chunks(chunks, empty_factory):
    if not chunks:
        return empty_factory()
    return np.concatenate(chunks, axis=0)


def load_train_val_split(all_files):
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []

    print("Zpracovavam train/val soubory a delim separatne...")
    gap = SEQUENCE_LENGTH

    for file_path in all_files:
        try:
            x_w, y_w = create_windows_for_file(file_path)
            n_samples = len(x_w)
            if n_samples == 0:
                continue

            idx_train_end = int(n_samples * TRAIN_SPLIT)
            val_start = idx_train_end + gap

            x_tr = x_w[:idx_train_end]
            y_tr = y_w[:idx_train_end]
            x_v = x_w[val_start:]
            y_v = y_w[val_start:]

            print(f"Processed train/val file {file_path}:")
            print(f"  Train samples: {len(x_tr)}")
            print(f"  Val samples:   {len(x_v)}")

            if len(x_tr) > 0:
                train_X_chunks.append(x_tr)
                train_y_chunks.append(y_tr)
            if len(x_v) > 0:
                val_X_chunks.append(x_v)
                val_y_chunks.append(y_v)

        except (ValueError, OSError, KeyError) as e:
            print(f"Skipping train/val file {file_path}: {e}")

    X_train = concat_chunks(train_X_chunks, empty_x)
    y_train = concat_chunks(train_y_chunks, empty_y)
    X_val = concat_chunks(val_X_chunks, empty_x)
    y_val = concat_chunks(val_y_chunks, empty_y)

    return (X_train, y_train), (X_val, y_val)


def safe_dataset_name(file_path):
    rel = Path(file_path)
    try:
        rel = rel.relative_to(EXPERIMENT_DIR)
    except ValueError:
        pass

    name = rel.as_posix()
    name = re.sub(r"\.jsonl$", "", name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "__", name)
    return name


def to_tensor_dataset(X_data, y_data):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def save_separate_test_datasets(test_files, scaler):
    Path(TEST_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    manifest = []

    for file_path in test_files:
        try:
            X_test_u, y_test_u = create_windows_for_file(file_path)
            if len(X_test_u) == 0:
                print(f"Skipping empty test dataset {file_path}")
                continue

            X_test, y_test = scale_data(X_test_u, y_test_u, scaler)
            dataset = to_tensor_dataset(X_test, y_test)

            dataset_name = safe_dataset_name(file_path)
            dataset_path = f"{TEST_DATASET_DIR}{dataset_name}{DATASET_POSTFIX}"
            torch.save(dataset, dataset_path)

            manifest.append({
                "source_file": file_path,
                "dataset_path": dataset_path,
                "samples": len(dataset),
            })

            print(f"Saved test dataset {dataset_path}: {len(dataset)} samples")

        except (ValueError, OSError, KeyError) as e:
            print(f"Skipping test file {file_path}: {e}")

    with open(TEST_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved test manifest to {TEST_MANIFEST_PATH}")
    return manifest


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    root_dir = join("../", EXPERIMENT_DIR)

    train_val_files = resolve_experiment_files(root_dir, TRAIN_VAL_EXPERIMENTS)
    test_files = resolve_experiment_files(root_dir, TEST_EXPERIMENTS)

    print(f"Found {len(train_val_files)} train/val dataset files in {root_dir}.")
    for file_path in train_val_files:
        print(f"  train/val: {file_path}")

    print(f"Found {len(test_files)} separate test dataset files in {root_dir}.")
    for file_path in test_files:
        print(f"  test: {file_path}")

    (X_train_u, y_train_u), (X_val_u, y_val_u) = load_train_val_split(train_val_files)

    print(f"Train samples: {len(X_train_u)}")
    print(f"Val samples:   {len(X_val_u)}")

    if len(X_train_u) == 0:
        raise ValueError("No training samples were generated from the selected train/val dataset files.")

    perm = np.random.permutation(len(X_train_u))
    X_train_u = X_train_u[perm]
    y_train_u = y_train_u[perm]

    print("Generating and scaling train/val dataset...")
    scaler = fit_scalers(X_train_u, y_train_u)

    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print("Scalers saved.")

    X_train, y_train = scale_data(X_train_u, y_train_u, scaler)
    X_val, y_val = scale_data(X_val_u, y_val_u, scaler)

    train_dataset = to_tensor_dataset(X_train, y_train)
    val_dataset = to_tensor_dataset(X_val, y_val)

    if not exists(DATASET_DIR):
        makedirs(DATASET_DIR)

    torch.save(train_dataset, f"{DATASET_DIR}train{DATASET_POSTFIX}")
    torch.save(val_dataset, f"{DATASET_DIR}val{DATASET_POSTFIX}")
    print(f"Train/val datasets saved to {DATASET_DIR}.")

    manifest = save_separate_test_datasets(test_files, scaler)
    print(f"Separate test datasets: {len(manifest)}")


if __name__ == "__main__":
    main()
