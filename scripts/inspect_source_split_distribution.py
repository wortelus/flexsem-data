"""Inspect step and experiment distributions used by the RNN dataset split.

Unlike ``inspect_split_distribution.py``, this script reads the configured raw
JSONL sources.  That lets it retain the source experiment/file for every
generated window while reproducing both split modes from
``rnn/run_1_generate_dataset.py``: hybrid per-trajectory splitting for single
files and whole-file splitting for directories containing multiple JSONL
files.

Only the ground-truth motor command delta associated with the final row of each
window is counted as its step. Consequently, overlapping history does not count
the same command many times.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows PowerShell commonly starts Python with a legacy console encoding that
# cannot print labels such as "≥".  CSV/PNG output is UTF-8 regardless, and this
# keeps the console report readable as well.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from rnn.run_1_generate_dataset import (  # noqa: E402
    EXPERIMENTS,
    prepare_source_split_plan,
    resolve_experiment_sources,
)
from rnn.utils.const import (  # noqa: E402
    DATASET_DIR_PATH,
    EXPERIMENT_DIR,
    INPUT_SIZE,
    INVERSE_MODEL,
    RUN_DIR,
    TARGET_MODE,
    WINDOW_COORD_MODE,
)


SPLITS = ("train", "val", "test")
SPLIT_COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test": "#54A24B",
}

# Intervals are left-closed/right-open, for example 3–5 µm means
# 3_000 <= command magnitude < 5_000 nm.  Keep the edges contiguous so every
# generated window belongs to exactly one bin and distribution totals remain
# equal to the split sizes.
STEP_EDGES_NM = np.array(
    [
        0.0,
        100.0,
        300.0,
        1_000.0,
        3_000.0,
        5_000.0,
        7_000.0,
        10_000.0,
        13_000.0,
        16_000.0,
        20_000.0,
        25_000.0,
        np.inf,
    ]
)
STEP_LABELS = (
    "0–100",
    "100–300",
    "300–1000",
    "1–3 µm",
    "3–5 µm",
    "5–7 µm",
    "8–10 µm",
    "10–13 µm",
    "13–16 µm",
    "16–20 µm",
    "20–25 µm",
    "≥25 µm",
)
FINE_STEP_BIN_COUNT = 30


def parse_args() -> argparse.Namespace:
    default_output = RUN_DIR / "plots" / "source_split_distribution"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for CSV and PNG output (default: {default_output})",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Do not compare reconstructed split sizes with saved train/val/test.pt.",
    )
    return parser.parse_args()


def experiment_name(experiment_spec: str) -> str:
    """Return the top-level run directory used as a logical experiment name."""
    return Path(experiment_spec).parts[0]


def classify_steps(step_size_nm: np.ndarray) -> pd.Categorical:
    return pd.cut(
        step_size_nm,
        bins=STEP_EDGES_NM,
        labels=STEP_LABELS,
        right=False,
        include_lowest=True,
        ordered=True,
    )


def collect_window_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    if INPUT_SIZE != 4:
        raise ValueError(
            "This inspection is defined for the current INPUT_SIZE=4 double "
            f"preprocessing, got INPUT_SIZE={INPUT_SIZE}."
        )

    raw_root = REPO_ROOT / EXPERIMENT_DIR
    records: list[dict[str, object]] = []
    file_records: list[dict[str, object]] = []
    seen_files: set[Path] = set()

    print("Configured raw experiments:")
    for experiment_spec in EXPERIMENTS:
        name = experiment_name(experiment_spec)
        source = resolve_experiment_sources(str(raw_root), [experiment_spec])[0]
        for file_name in source["files"]:
            file_path = Path(file_name).resolve()
            if file_path in seen_files:
                raise ValueError(f"Source file is configured more than once: {file_path}")
            seen_files.add(file_path)

        # The generator owns all split decisions. Suppressing its per-windowing
        # status lines keeps this report focused on distributions.
        with contextlib.redirect_stdout(io.StringIO()):
            plans, skipped_units = prepare_source_split_plan(source)

        used_files = {
            Path(plan["chunk"]["file_path"]).resolve()
            for plan in plans
        }

        for skipped in skipped_units:
            skipped_row = dict(skipped)
            skipped_row.pop("source", None)
            source_file = Path(str(skipped_row["source_file"]))
            if source_file.parts and source_file.parts[0] == EXPERIMENT_DIR:
                source_file = Path(*source_file.parts[1:])
            skipped_row["source_file"] = source_file.as_posix()
            skipped_row["experiment"] = name
            file_records.append(skipped_row)

        for plan in plans:
            chunk = plan["chunk"]
            file_path = Path(chunk["file_path"]).resolve()
            relative_file = file_path.relative_to(raw_root.resolve()).as_posix()
            x_windows = chunk["x"]
            y_windows = chunk["y"]
            split_indices = plan["split_indices"]

            file_row: dict[str, object] = {
                "experiment": name,
                "source_file": relative_file,
                "unit_type": chunk["unit_type"],
                "segment_index": chunk.get("segment_index"),
                "segment_label": chunk.get("segment_label"),
                "segment_raw_start_idx": chunk.get("raw_start_idx"),
                "segment_raw_end_idx": chunk.get("raw_end_idx"),
                "status": "used",
                "skip_reason": "",
                "split_mode": plan["split_mode"],
                "assigned_split": plan["assigned_split"],
                "raw_rows": chunk["raw_rows"],
                "generated_windows": len(x_windows),
            }

            for split, indices in split_indices.items():
                file_row[f"{split}_windows"] = len(indices)
                if len(indices) == 0:
                    continue

                selected = x_windows[indices]
                selected_y = y_windows[indices]

                if WINDOW_COORD_MODE != "delta":
                    raise ValueError(
                        "Command-step inspection requires WINDOW_COORD_MODE='delta'."
                    )

                input_delta_nm = selected[:, -1, :2]
                if INVERSE_MODEL:
                    if TARGET_MODE == "residual_delta":
                        # inverse residual = command_delta - actual_delta
                        step_xy_nm = input_delta_nm + selected_y
                    elif TARGET_MODE == "actual_delta":
                        step_xy_nm = selected_y
                    else:
                        raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
                else:
                    # Forward delta input is the motor command delta.
                    step_xy_nm = input_delta_nm

                step_xy_nm = np.rint(step_xy_nm)
                step_size_nm = np.linalg.norm(step_xy_nm, axis=1)
                step_bins = classify_steps(step_size_nm)

                for window_index, step_xy, step_size, step_bin in zip(
                    indices, step_xy_nm, step_size_nm, step_bins
                ):
                    records.append(
                        {
                            "experiment": name,
                            "source_file": relative_file,
                            "segment_index": chunk.get("segment_index"),
                            "segment_label": chunk.get("segment_label"),
                            "split": split,
                            "window_index_in_generated_file": int(window_index),
                            "raw_window_start_idx": (
                                int(chunk.get("raw_start_idx", 0))
                                + int(window_index)
                            ),
                            "command_dx_nm": float(step_xy[0]),
                            "command_dy_nm": float(step_xy[1]),
                            "command_size_nm": float(step_size),
                            "step_bin": str(step_bin),
                        }
                    )

            file_records.append(file_row)

        print(
            f"  {name}: {len(source['files'])} JSONL, "
            f"used_files={len(used_files)}, split_units={len(plans)}, "
            f"skipped_units={len(skipped_units)}"
        )

    if not records:
        raise ValueError("No split windows were reconstructed from the configured sources.")

    windows = pd.DataFrame.from_records(records)
    windows["split"] = pd.Categorical(windows["split"], SPLITS, ordered=True)
    windows["step_bin"] = pd.Categorical(
        windows["step_bin"], STEP_LABELS, ordered=True
    )

    files = pd.DataFrame.from_records(file_records)
    return windows, files


def step_distribution_by_split(windows: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product(
        [SPLITS, STEP_LABELS], names=["split", "step_bin"]
    )
    result = (
        windows.groupby(["split", "step_bin"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = result.groupby("split")["count"].transform("sum")
    result["share_within_split_pct"] = np.where(
        totals > 0, 100.0 * result["count"] / totals, 0.0
    )
    return result


def experiment_distribution_by_split(windows: pd.DataFrame) -> pd.DataFrame:
    experiments = list(dict.fromkeys(windows["experiment"].astype(str)))
    full_index = pd.MultiIndex.from_product(
        [experiments, SPLITS], names=["experiment", "split"]
    )
    result = (
        windows.groupby(["experiment", "split"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    split_totals = result.groupby("split")["count"].transform("sum")
    experiment_totals = result.groupby("experiment")["count"].transform("sum")
    result["share_within_split_pct"] = np.where(
        split_totals > 0, 100.0 * result["count"] / split_totals, 0.0
    )
    result["share_within_experiment_pct"] = np.where(
        experiment_totals > 0, 100.0 * result["count"] / experiment_totals, 0.0
    )
    return result


def step_distribution_by_experiment_and_split(windows: pd.DataFrame) -> pd.DataFrame:
    experiments = list(dict.fromkeys(windows["experiment"].astype(str)))
    full_index = pd.MultiIndex.from_product(
        [experiments, SPLITS, STEP_LABELS],
        names=["experiment", "split", "step_bin"],
    )
    result = (
        windows.groupby(["experiment", "split", "step_bin"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = result.groupby(["experiment", "split"])["count"].transform("sum")
    result["share_within_experiment_split_pct"] = np.where(
        totals > 0, 100.0 * result["count"] / totals, 0.0
    )
    return result


def print_step_report(distribution: pd.DataFrame) -> None:
    print("\nCOMMAND-STEP DISTRIBUTION BY SPLIT")
    for split in SPLITS:
        rows = distribution[distribution["split"] == split]
        total = int(rows["count"].sum())
        print(f"\n{split}: {total} windows")
        for row in rows.itertuples(index=False):
            print(
                f"  {str(row.step_bin):10} "
                f"{int(row.count):7d}  {row.share_within_split_pct:6.2f} %"
            )


def print_experiment_report(
    experiment_distribution: pd.DataFrame,
    experiment_step_distribution: pd.DataFrame,
) -> None:
    counts = experiment_distribution.pivot(
        index="experiment", columns="split", values="count"
    ).reindex(columns=SPLITS, fill_value=0)
    counts["total"] = counts.sum(axis=1)

    print("\nWINDOW DISTRIBUTION BY EXPERIMENT AND SPLIT")
    print(counts.to_string())

    all_steps = (
        experiment_step_distribution.groupby(
            ["experiment", "step_bin"], observed=True
        )["count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=STEP_LABELS, fill_value=0)
    )
    row_totals = all_steps.sum(axis=1)

    formatted = all_steps.copy().astype(object)
    for experiment in all_steps.index:
        for step_bin in all_steps.columns:
            count = int(all_steps.loc[experiment, step_bin])
            share = 100.0 * count / row_totals.loc[experiment]
            formatted.loc[experiment, step_bin] = f"{count} ({share:.1f}%)"

    print("\nCOMMAND-STEP DISTRIBUTION WITHIN EACH EXPERIMENT (ALL SPLITS)")
    print(formatted.to_string())


def verify_saved_split_sizes(windows: pd.DataFrame) -> None:
    print("\nSAVED DATASET SIZE CHECK")
    for split in SPLITS:
        dataset_path = DATASET_DIR_PATH / f"{split}.pt"
        reconstructed = int((windows["split"] == split).sum())
        if not dataset_path.exists():
            print(f"  {split}: {dataset_path} does not exist")
            continue

        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        saved = len(dataset)
        status = "OK" if saved == reconstructed else "MISMATCH"
        print(
            f"  {split}: reconstructed={reconstructed}, "
            f"saved={saved}  [{status}]"
        )


def plot_step_distribution_by_split(
    distribution: pd.DataFrame, output_path: Path
) -> None:
    fig_width = max(12.0, 0.95 * len(STEP_LABELS))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    x = np.arange(len(STEP_LABELS))
    width = 0.24

    for offset, split in enumerate(SPLITS):
        rows = distribution[distribution["split"] == split].set_index("step_bin")
        values = rows.reindex(STEP_LABELS)["share_within_split_pct"].to_numpy()
        positions = x + (offset - 1) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=split,
            color=SPLIT_COLORS[split],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)

    ax.set_title("Command-step distribution by dataset split")
    ax.set_xlabel("Ground-truth command magnitude")
    ax.set_ylabel("Share within split [%]")
    ax.set_xticks(x, STEP_LABELS, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_distribution_by_split_fine(
    windows: pd.DataFrame,
    output_path: Path,
    bin_count: int = FINE_STEP_BIN_COUNT,
) -> None:
    """Plot vertically aligned split histograms using shared equal-width bins."""
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")

    all_step_sizes_nm = windows["command_size_nm"].to_numpy(dtype=float)
    if not np.all(np.isfinite(all_step_sizes_nm)):
        raise ValueError("Command-step magnitudes contain NaN or infinite values")
    if len(all_step_sizes_nm) == 0:
        raise ValueError("Cannot plot an empty command-step distribution")

    # Round the common upper bound up to a whole micrometre.  Every subplot
    # consequently uses the exact same physical range and equal-width bins.
    max_step_nm = float(np.max(all_step_sizes_nm))
    upper_bound_nm = max(1_000.0, np.ceil(max_step_nm / 1_000.0) * 1_000.0)
    bin_edges_nm = np.linspace(0.0, upper_bound_nm, bin_count + 1)
    bin_width_nm = bin_edges_nm[1] - bin_edges_nm[0]
    tick_labels_nm = [
        f"{edge_nm:,.0f}".replace(",", " ") for edge_nm in bin_edges_nm
    ]

    fig, axes = plt.subplots(
        len(SPLITS),
        1,
        figsize=(20, 14),
        sharex=True,
        sharey=True,
    )
    maximum_share = 0.0

    for ax, split in zip(axes, SPLITS):
        split_sizes_nm = windows.loc[
            windows["split"] == split, "command_size_nm"
        ].to_numpy(dtype=float)
        counts, _ = np.histogram(split_sizes_nm, bins=bin_edges_nm)
        shares = (
            100.0 * counts / len(split_sizes_nm)
            if len(split_sizes_nm)
            else np.zeros(bin_count, dtype=float)
        )
        exact_zero_count = int(np.count_nonzero(np.abs(split_sizes_nm) < 0.5))
        exact_zero_share = (
            100.0 * exact_zero_count / len(split_sizes_nm)
            if len(split_sizes_nm)
            else 0.0
        )
        maximum_share = max(maximum_share, float(np.max(shares, initial=0.0)))

        ax.bar(
            bin_edges_nm[:-1],
            shares,
            width=np.diff(bin_edges_nm),
            align="edge",
            color=SPLIT_COLORS[split],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(
            f"{split} (n={len(split_sizes_nm)}; exact 0 nm: "
            f"{exact_zero_count} = {exact_zero_share:.2f}%)",
            loc="left",
        )
        ax.set_ylabel("Share [%]")
        ax.set_xlabel("Ground-truth command magnitude [nm]")
        ax.set_xticks(bin_edges_nm)
        ax.set_xticklabels(
            tick_labels_nm,
            rotation=55,
            ha="right",
            fontsize=7,
        )
        # Matplotlib hides upper x tick labels for shared axes by default.
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(axis="y", alpha=0.25)

    shared_y_max = max(1.0, maximum_share * 1.12)
    axes[0].set_ylim(0.0, shared_y_max)
    axes[-1].set_xlim(0.0, upper_bound_nm)
    fig.suptitle(
        "Command-step distribution by split — "
        f"{bin_count} shared equal-width bins "
        f"(0–{upper_bound_nm:,.0f} nm, width {bin_width_nm:,.0f} nm)".replace(
            ",", " "
        )
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_experiment_distribution(
    distribution: pd.DataFrame, output_path: Path
) -> None:
    counts = distribution.pivot(
        index="experiment", columns="split", values="count"
    ).reindex(columns=SPLITS, fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]

    fig_height = max(6.0, 0.55 * len(counts))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    left = np.zeros(len(counts))
    y = np.arange(len(counts))

    for split in SPLITS:
        values = counts[split].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            label=split,
            color=SPLIT_COLORS[split],
        )
        left += values

    for row, total in enumerate(left):
        ax.text(total, row, f"  {int(total)}", va="center", fontsize=8)

    ax.set_title("Generated windows by experiment and split")
    ax.set_xlabel("Number of windows")
    ax.set_yticks(y, counts.index)
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def percentage_heatmap(
    ax: plt.Axes,
    values: pd.DataFrame,
    title: str,
    show_y_labels: bool = True,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(values.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(values.columns)), values.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(values.index)))
    if show_y_labels:
        ax.set_yticklabels(values.index)
    else:
        ax.tick_params(axis="y", labelleft=False)

    for row in range(len(values.index)):
        for column in range(len(values.columns)):
            value = values.iat[row, column]
            text_color = "white" if value >= 55 else "black"
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    return image


def plot_experiment_step_heatmap(
    windows: pd.DataFrame, output_path: Path
) -> None:
    counts = (
        windows.groupby(["experiment", "step_bin"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STEP_LABELS, fill_value=0)
    )
    percentages = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100.0

    fig_height = max(6.0, 0.55 * len(percentages))
    fig_width = max(12.0, 0.9 * len(STEP_LABELS))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = percentage_heatmap(
        ax,
        percentages,
        "Command-step distribution within each experiment (all splits) [%]",
    )
    fig.colorbar(image, ax=ax, label="Share within experiment [%]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_experiment_step_heatmaps_by_split(
    windows: pd.DataFrame, output_path: Path
) -> None:
    experiments = list(dict.fromkeys(windows["experiment"].astype(str)))
    fig_height = max(6.0, 0.55 * len(experiments))
    fig_width = max(24.0, 1.9 * len(STEP_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    last_image = None

    for index, (ax, split) in enumerate(zip(axes, SPLITS)):
        selected = windows[windows["split"] == split]
        counts = (
            selected.groupby(["experiment", "step_bin"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=experiments, columns=STEP_LABELS, fill_value=0)
        )
        percentages = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100.0
        last_image = percentage_heatmap(
            ax,
            percentages,
            f"{split} [%]",
            show_y_labels=index == 0,
        )

    assert last_image is not None
    fig.suptitle("Command-step distribution within experiment and split")
    fig.subplots_adjust(left=0.25, right=0.88, bottom=0.16, top=0.88, wspace=0.08)
    colorbar_axis = fig.add_axes((0.91, 0.20, 0.015, 0.60))
    fig.colorbar(
        last_image,
        cax=colorbar_axis,
        label="Share within experiment/split [%]",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_outputs(
    windows: pd.DataFrame,
    files: pd.DataFrame,
    step_distribution: pd.DataFrame,
    experiment_distribution: pd.DataFrame,
    experiment_step_distribution: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    windows.to_csv(output_dir / "window_step_metadata.csv", index=False)
    files.to_csv(output_dir / "source_file_distribution.csv", index=False)
    step_distribution.to_csv(
        output_dir / "step_distribution_by_split.csv", index=False
    )
    experiment_distribution.to_csv(
        output_dir / "experiment_distribution_by_split.csv", index=False
    )
    experiment_step_distribution.to_csv(
        output_dir / "step_distribution_by_experiment_and_split.csv", index=False
    )

    plot_step_distribution_by_split(
        step_distribution, output_dir / "step_distribution_by_split.png"
    )
    plot_step_distribution_by_split_fine(
        windows,
        output_dir / "step_distribution_by_split_30_bins.png",
    )
    plot_experiment_distribution(
        experiment_distribution,
        output_dir / "experiment_distribution_by_split.png",
    )
    plot_experiment_step_heatmap(
        windows, output_dir / "step_distribution_by_experiment.png"
    )
    plot_experiment_step_heatmaps_by_split(
        windows,
        output_dir / "step_distribution_by_experiment_and_split.png",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    windows, files = collect_window_metadata()
    step_distribution = step_distribution_by_split(windows)
    experiment_distribution = experiment_distribution_by_split(windows)
    experiment_step_distribution = step_distribution_by_experiment_and_split(windows)

    print_step_report(step_distribution)
    print_experiment_report(experiment_distribution, experiment_step_distribution)

    if not args.skip_dataset_check:
        verify_saved_split_sizes(windows)

    save_outputs(
        windows,
        files,
        step_distribution,
        experiment_distribution,
        experiment_step_distribution,
        output_dir,
    )

    print(f"\nSaved CSV tables and PNG plots to:\n  {output_dir}")


if __name__ == "__main__":
    main()
