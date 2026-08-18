import hashlib
import json
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
    from rnn.preprocessing.single import (
        _iter_segments as iter_trajectory_segments,
        create_windows,
        fit_scalers,
        scale_data,
    )

    print(f"Using preprocessing for INPUT_SIZE = 2")
elif INPUT_SIZE == 4:
    from rnn.preprocessing.double import (
        _iter_segments as iter_trajectory_segments,
        create_windows,
        fit_scalers,
        scale_data,
    )

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
    "run84_random_walk",
]


def resolve_experiment_sources(root_dir, experiments=None):
    """Resolve configured sources without losing directory boundaries.

    A directory containing multiple JSONL files is treated as one logical
    dataset whose files are assigned whole to train/val/test. A direct JSONL
    path, or a directory containing only one JSONL file, is split separately
    by each continuous trajectory segment inside the file.
    """
    if not experiments:
        experiments = [str(path) for path in sorted(Path(root_dir).glob("*.jsonl"))]

    sources = []
    seen_files = set()

    for experiment in experiments:
        path = Path(experiment)
        if not path.is_absolute():
            path = Path(root_dir) / path

        if path.is_dir():
            files = sorted(str(file_path) for file_path in path.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(
                    f"Experiment directory contains no .jsonl files: {path}"
                )
            split_mode = "file_level" if len(files) > 1 else "segment_level"
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            files = [str(path)]
            split_mode = "segment_level"
        else:
            raise FileNotFoundError(f"Experiment path is not a .jsonl file or directory: {path}")

        for file_path in files:
            resolved_file = str(Path(file_path).resolve())
            if resolved_file in seen_files:
                raise ValueError(f"Dataset file is configured more than once: {file_path}")
            seen_files.add(resolved_file)

        sources.append(
            {
                "experiment": str(experiment),
                "path": str(path),
                "split_mode": split_mode,
                "files": files,
            }
        )

    return sources


def resolve_experiment_files(root_dir, experiments=None):
    """Backward-compatible flattened resolver used by the custom generator."""
    sources = resolve_experiment_sources(root_dir, experiments)
    return [file_path for source in sources for file_path in source["files"]]


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


SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "test": TEST_SPLIT,
}


def split_window_indices(number_of_windows):
    """Return the historical chronological split with leakage guard gaps."""
    train_end = int(number_of_windows * TRAIN_SPLIT)
    val_end = int(number_of_windows * (TRAIN_SPLIT + VAL_SPLIT))
    gap = SEQUENCE_LENGTH

    val_start = train_end + gap if VAL_SPLIT > 0 else val_end
    test_start = val_end + gap if TEST_SPLIT > 0 else number_of_windows

    return {
        "train": np.arange(0, train_end, dtype=np.int64),
        "val": np.arange(val_start, val_end, dtype=np.int64)
        if VAL_SPLIT > 0
        else np.empty(0, dtype=np.int64),
        "test": np.arange(test_start, number_of_windows, dtype=np.int64)
        if TEST_SPLIT > 0
        else np.empty(0, dtype=np.int64),
    }


def assign_groups_to_splits(
    group_chunks,
    source_name,
    initial_window_counts=None,
    group_count_weight=0.1,
    normalize_by_split_target=True,
):
    """Assign every prepared group wholly to one split.

    Window count is the main balance target and group count is secondary. For
    hybrid segment splitting, ``initial_window_counts`` contains windows that
    were already assigned by chronological splits of long trajectories.
    """
    active_splits = [name for name in SPLIT_NAMES if SPLIT_FRACTIONS[name] > 0]
    if not active_splits:
        raise ValueError("At least one dataset split must have a positive fraction")

    if initial_window_counts is None:
        initial_window_counts = {split: 0 for split in SPLIT_NAMES}

    total_windows = sum(initial_window_counts.values()) + sum(
        len(chunk["x"]) for chunk in group_chunks
    )
    targets = {
        split: total_windows * SPLIT_FRACTIONS[split]
        for split in active_splits
    }
    group_targets = {
        split: len(group_chunks) * SPLIT_FRACTIONS[split]
        for split in active_splits
    }
    counts = {
        split: int(initial_window_counts.get(split, 0))
        for split in active_splits
    }
    group_counts = {split: 0 for split in active_splits}

    source_digest = int.from_bytes(
        hashlib.sha256(source_name.encode("utf-8")).digest()[:4],
        byteorder="little",
    )
    rng = np.random.default_rng(np.random.SeedSequence([SEED, source_digest]))
    random_tie_break = {
        chunk["group_id"]: float(rng.random()) for chunk in group_chunks
    }
    split_priority = {
        split: priority
        for priority, split in enumerate(rng.permutation(active_splits))
    }

    ordered_chunks = sorted(
        group_chunks,
        key=lambda chunk: (
            -len(chunk["x"]),
            random_tie_break[chunk["group_id"]],
        ),
    )
    assignments = {}

    for chunk in ordered_chunks:
        weight = len(chunk["x"])

        def assignment_score(candidate_split):
            candidate_counts = counts.copy()
            candidate_counts[candidate_split] += weight
            candidate_group_counts = group_counts.copy()
            candidate_group_counts[candidate_split] += 1
            if normalize_by_split_target:
                window_error = sum(
                    (
                        (candidate_counts[split] - targets[split])
                        / max(targets[split], 1.0)
                    ) ** 2
                    for split in active_splits
                )
                group_count_error = sum(
                    (
                        (candidate_group_counts[split] - group_targets[split])
                        / max(group_targets[split], 1.0)
                    ) ** 2
                    for split in active_splits
                )
            else:
                window_error = sum(
                    (candidate_counts[split] - targets[split]) ** 2
                    for split in active_splits
                ) / max(total_windows, 1) ** 2
                group_count_error = sum(
                    (candidate_group_counts[split] - group_targets[split]) ** 2
                    for split in active_splits
                ) / max(len(group_chunks), 1) ** 2
            # Window count is the primary balance target. The smaller group-count
            # term avoids representing a split with one unusually long group when
            # similarly accurate multi-file assignments are available.
            return (
                window_error + group_count_weight * group_count_error,
                split_priority[candidate_split],
            )

        chosen_split = min(active_splits, key=assignment_score)
        assignments[chunk["group_id"]] = chosen_split
        counts[chosen_split] += weight
        group_counts[chosen_split] += 1

    return assignments


def assign_files_to_splits(file_chunks, source_name):
    """Backward-compatible whole-file assignment helper."""
    groups = []
    for chunk in file_chunks:
        group = dict(chunk)
        group["group_id"] = chunk["file_path"]
        groups.append(group)
    return assign_groups_to_splits(groups, source_name)


def _manifest_file_path(file_path):
    path = Path(file_path).resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _segment_label(segment, segment_index):
    parts = [f"segment_{segment_index:03d}"]
    if "experiment_name" in segment.columns:
        parts.append(str(segment.iloc[0]["experiment_name"]))
    if "iteration" in segment.columns:
        parts.append(f"iteration={segment.iloc[0]['iteration']}")
    if "step" in segment.columns:
        parts.append(
            f"steps={segment.iloc[0]['step']}..{segment.iloc[-1]['step']}"
        )
    return " | ".join(parts)


def _load_file_window_chunk(file_path):
    df = load_dataset_file(file_path)
    report_dataset_sanity(df, file_path)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    x_windows, y_windows = create_windows(df, SEQUENCE_LENGTH)
    return {
        "group_id": str(Path(file_path).resolve()),
        "unit_type": "file",
        "file_path": file_path,
        "segment_index": None,
        "segment_label": None,
        "raw_start_idx": 0,
        "raw_end_idx": len(df) - 1,
        "raw_rows": len(df),
        "x": x_windows,
        "y": y_windows,
    }


def _load_segment_window_chunks(file_path):
    df = load_dataset_file(file_path)
    report_dataset_sanity(df, file_path)
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    chunks = []
    skipped = []
    raw_offset = 0

    for segment_index, segment in enumerate(iter_trajectory_segments(df), 1):
        raw_start_idx = raw_offset
        raw_end_idx = raw_offset + len(segment) - 1
        raw_offset += len(segment)
        segment_label = _segment_label(segment, segment_index)
        group_id = f"{Path(file_path).resolve()}::segment:{segment_index}"

        try:
            x_windows, y_windows = create_windows(segment, SEQUENCE_LENGTH)
        except (ValueError, KeyError) as exc:
            skipped.append(
                {
                    "group_id": group_id,
                    "unit_type": "segment",
                    "file_path": file_path,
                    "segment_index": segment_index,
                    "segment_label": segment_label,
                    "raw_start_idx": raw_start_idx,
                    "raw_end_idx": raw_end_idx,
                    "raw_rows": len(segment),
                    "skip_reason": str(exc),
                }
            )
            continue

        chunks.append(
            {
                "group_id": group_id,
                "unit_type": "segment",
                "file_path": file_path,
                "segment_index": segment_index,
                "segment_label": segment_label,
                "raw_start_idx": raw_start_idx,
                "raw_end_idx": raw_end_idx,
                "raw_rows": len(segment),
                "x": x_windows,
                "y": y_windows,
            }
        )

    return chunks, skipped


def _empty_split_indices():
    return {
        split: np.empty(0, dtype=np.int64)
        for split in SPLIT_NAMES
    }


def _whole_group_split_indices(number_of_windows, assigned_split):
    result = _empty_split_indices()
    result[assigned_split] = np.arange(number_of_windows, dtype=np.int64)
    return result


def _can_use_chronological_split(split_indices):
    return all(
        SPLIT_FRACTIONS[split] == 0 or len(split_indices[split]) > 0
        for split in SPLIT_NAMES
    )


def _skipped_manifest_row(source, unit, split_mode):
    return {
        "source": source["experiment"],
        "source_file": _manifest_file_path(unit["file_path"]),
        "unit_type": unit.get("unit_type", "file"),
        "segment_index": unit.get("segment_index"),
        "segment_label": unit.get("segment_label"),
        "segment_raw_start_idx": unit.get("raw_start_idx"),
        "segment_raw_end_idx": unit.get("raw_end_idx"),
        "split_mode": split_mode,
        "assigned_split": None,
        "status": "skipped",
        "skip_reason": unit["skip_reason"],
        "raw_rows": unit.get("raw_rows", 0),
        "generated_windows": 0,
        "train_windows": 0,
        "val_windows": 0,
        "test_windows": 0,
    }


def _planned_manifest_row(source, plan):
    chunk = plan["chunk"]
    split_counts = {
        split: len(indices)
        for split, indices in plan["split_indices"].items()
    }
    return {
        "source": source["experiment"],
        "source_file": _manifest_file_path(chunk["file_path"]),
        "unit_type": chunk["unit_type"],
        "segment_index": chunk.get("segment_index"),
        "segment_label": chunk.get("segment_label"),
        "segment_raw_start_idx": chunk.get("raw_start_idx"),
        "segment_raw_end_idx": chunk.get("raw_end_idx"),
        "split_mode": plan["split_mode"],
        "assigned_split": plan["assigned_split"],
        "status": "used",
        "skip_reason": "",
        "raw_rows": chunk["raw_rows"],
        "generated_windows": len(chunk["x"]),
        "train_windows": split_counts["train"],
        "val_windows": split_counts["val"],
        "test_windows": split_counts["test"],
    }


def prepare_source_split_plan(source):
    """Create window chunks and their split indices for one configured source."""
    plans = []
    skipped_manifest = []

    if source["split_mode"] == "file_level":
        chunks = []
        for file_path in source["files"]:
            try:
                chunks.append(_load_file_window_chunk(file_path))
            except (ValueError, OSError, KeyError) as exc:
                skipped_manifest.append(
                    _skipped_manifest_row(
                        source,
                        {
                            "file_path": file_path,
                            "skip_reason": str(exc),
                        },
                        "file_level",
                    )
                )

        if chunks:
            assignments = assign_groups_to_splits(chunks, source["experiment"])
            for chunk in chunks:
                assigned_split = assignments[chunk["group_id"]]
                plans.append(
                    {
                        "chunk": chunk,
                        "split_mode": "file_level",
                        "assigned_split": assigned_split,
                        "split_indices": _whole_group_split_indices(
                            len(chunk["x"]), assigned_split
                        ),
                    }
                )
        return plans, skipped_manifest

    segment_chunks = []
    for file_path in source["files"]:
        try:
            chunks, skipped_segments = _load_segment_window_chunks(file_path)
            segment_chunks.extend(chunks)
            skipped_manifest.extend(
                _skipped_manifest_row(source, unit, "segment_level")
                for unit in skipped_segments
            )
        except (ValueError, OSError, KeyError) as exc:
            skipped_manifest.append(
                _skipped_manifest_row(
                    source,
                    {
                        "file_path": file_path,
                        "skip_reason": str(exc),
                    },
                    "segment_level",
                )
            )

    chronological_plans = []
    short_chunks = []
    initial_window_counts = {split: 0 for split in SPLIT_NAMES}

    for chunk in segment_chunks:
        split_indices = split_window_indices(len(chunk["x"]))
        if _can_use_chronological_split(split_indices):
            plan = {
                "chunk": chunk,
                "split_mode": "within_segment",
                "assigned_split": "chronological",
                "split_indices": split_indices,
            }
            chronological_plans.append(plan)
            for split, indices in split_indices.items():
                initial_window_counts[split] += len(indices)
        else:
            short_chunks.append(chunk)

    plans.extend(chronological_plans)

    if short_chunks:
        assignments = assign_groups_to_splits(
            short_chunks,
            f"{source['experiment']}::short_segments",
            initial_window_counts=initial_window_counts,
            group_count_weight=0.0,
            normalize_by_split_target=False,
        )
        for chunk in short_chunks:
            assigned_split = assignments[chunk["group_id"]]
            plans.append(
                {
                    "chunk": chunk,
                    "split_mode": "whole_segment",
                    "assigned_split": assigned_split,
                    "split_indices": _whole_group_split_indices(
                        len(chunk["x"]), assigned_split
                    ),
                }
            )

    plans.sort(
        key=lambda plan: (
            str(plan["chunk"]["file_path"]),
            plan["chunk"].get("segment_index") or 0,
        )
    )
    return plans, skipped_manifest


def load_and_split_sources(sources):
    x_chunks = {split: [] for split in SPLIT_NAMES}
    y_chunks = {split: [] for split in SPLIT_NAMES}
    test_meta = []
    manifest_units = []

    def append_windows(split, chunk, indices):
        if len(indices) == 0:
            return
        x_chunks[split].append(chunk["x"][indices])
        y_chunks[split].append(chunk["y"][indices])
        if split == "test":
            for window_idx in indices:
                window_idx = int(window_idx)
                raw_start_idx = chunk.get("raw_start_idx", 0) + window_idx
                test_meta.append(
                    {
                        "source_file": chunk["file_path"],
                        "segment_index": chunk.get("segment_index"),
                        "segment_label": chunk.get("segment_label"),
                        "window_idx_in_file": raw_start_idx,
                        "raw_start_idx": raw_start_idx,
                        "raw_end_idx": raw_start_idx + SEQUENCE_LENGTH - 1,
                    }
                )

    print("Zpracovávám zdroje datasetu...")

    for source in sources:
        plans, skipped_manifest = prepare_source_split_plan(source)
        manifest_units.extend(skipped_manifest)
        for skipped in skipped_manifest:
            print(
                f"Skipping {skipped['source_file']}"
                f" segment={skipped['segment_index']}: {skipped['skip_reason']}"
            )

        if not plans:
            print(f"Source {source['experiment']}: no usable files")
            continue

        source_counts = {split: 0 for split in SPLIT_NAMES}
        source_unit_counts = {split: 0 for split in SPLIT_NAMES}
        mode_counts = {}

        for plan in plans:
            for split, indices in plan["split_indices"].items():
                append_windows(split, plan["chunk"], indices)
                source_counts[split] += len(indices)
                if len(indices) > 0:
                    source_unit_counts[split] += 1
            mode_counts[plan["split_mode"]] = (
                mode_counts.get(plan["split_mode"], 0) + 1
            )
            manifest_units.append(_planned_manifest_row(source, plan))

        print(
            f"Source {source['experiment']}: "
            + ", ".join(
                f"{mode}={count}" for mode, count in sorted(mode_counts.items())
            )
        )
        for split in SPLIT_NAMES:
            print(
                f"  {split.capitalize():5} units: {source_unit_counts[split]:3d}, "
                f"samples: {source_counts[split]:5d}"
            )

    def concat_x(chunks):
        if not chunks:
            return np.empty((0, SEQUENCE_LENGTH, INPUT_SIZE))
        return np.concatenate(chunks, axis=0)

    def concat_y(chunks):
        if not chunks:
            return np.empty((0, 2))
        return np.concatenate(chunks, axis=0)

    split_data = {
        split: (concat_x(x_chunks[split]), concat_y(y_chunks[split]))
        for split in SPLIT_NAMES
    }
    return (
        split_data["train"],
        split_data["val"],
        split_data["test"],
        test_meta,
        manifest_units,
    )


def load_and_split_per_file(all_files):
    """Compatibility wrapper using the segment-aware split for each file."""
    sources = [
        {
            "experiment": str(file_path),
            "path": str(file_path),
            "split_mode": "segment_level",
            "files": [str(file_path)],
        }
        for file_path in all_files
    ]
    train, val, test, test_meta, _ = load_and_split_sources(sources)
    return train, val, test, test_meta


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

    sources = resolve_experiment_sources(str(REPO_ROOT / EXPERIMENT_DIR), EXPERIMENTS)
    all_files = [file_path for source in sources for file_path in source["files"]]
    print(f"Found {len(all_files)} dataset files in {EXPERIMENT_DIR}.")
    for source in sources:
        print(
            f"  {source['experiment']}: {len(source['files'])} file(s), "
            f"split_mode={source['split_mode']}"
        )

    (
        (X_train_u, y_train_u),
        (X_val_u, y_val_u),
        (X_test_u, y_test_u),
        test_meta,
        manifest_units,
    ) = load_and_split_sources(sources)

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
    split_manifest = {
        "version": 2,
        "seed": SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "directory_split": (
            "Directories with multiple JSONL files are split by whole files, "
            "balanced primarily by generated-window count and secondarily by "
            "file count."
        ),
        "single_file_split": (
            "Direct JSONL files and one-file directories are split per "
            "continuous trajectory. Long trajectories use chronological "
            "splits with sequence-length guard gaps; short trajectories are "
            "assigned whole to one split."
        ),
        "total_windows": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "split_units": manifest_units,
    }
    split_manifest_path = Path(DATASET_DIR) / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Datasets saved to {DATASET_DIR} directory.")
    print(f"Split manifest saved to {split_manifest_path}.")

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
