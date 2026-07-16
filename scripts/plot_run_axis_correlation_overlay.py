import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data_original"
DEFAULT_OUTPUT = REPO_ROOT / "plots" / "run70_run71_run72_axis_correlation_overlay.png"

RUN_ORDER = ("run70", "run71", "run72")
RUN_COLORS = {
    "run70": "#1f77b4",
    "run71": "#2ca02c",
    "run72": "#d62728",
}

RUN_INPUTS = {
    "run70": [
        "run70-data-feast-overnight-sub0/confidence_0.8_no_axis_outliers_segments",
        "run70-data-feast-overnight-sub1/confidence_0.8_no_axis_outliers_segments",
        "run70-data-feast-overnight-sub2/confidence_0.8_no_axis_outliers_segments",
    ],
    "run71": [
        "run71-data-feast-overnight/confidence_0.7_segments",
    ],
    "run72": [
        "run72-data-feast-overnight/hysteresis_dataset_20260303_203815_updated.jsonl",
    ],
}

REQUIRED_COLUMNS = ("x_target_abs", "y_target_abs", "x_actual_abs", "y_actual_abs")


def load_data(path):
    """Load either JSON array/object or JSONL into a DataFrame."""
    with path.open("r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        return pd.DataFrame()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        return pd.DataFrame([parsed])
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line in {path}")
        return pd.DataFrame(records)


def expand_input_path(path):
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8-sig") as f:
                    manifest = json.load(f)
                segment_files = [
                    path / segment["file"]
                    for segment in manifest.get("segments", [])
                    if "file" in segment
                ]
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"WARNING: Could not read manifest {manifest_path}: {exc}")
                segment_files = []
        else:
            segment_files = sorted(path.glob("*.jsonl"))

        existing = [segment_path for segment_path in segment_files if segment_path.exists()]
        missing = [segment_path for segment_path in segment_files if not segment_path.exists()]
        if missing:
            print(f"WARNING: {path} has {len(missing)} segment(s) listed but missing.")
        if not existing:
            print(f"WARNING: Directory {path} contains no JSONL segment files.")
        return existing

    return [path]


def discover_files(data_root):
    discovered = {}
    for run_name in RUN_ORDER:
        run_paths = []
        for input_path in RUN_INPUTS.get(run_name, []):
            path = Path(input_path)
            if not path.is_absolute():
                path = data_root / path
            run_paths.extend(expand_input_path(path))
        discovered[run_name] = run_paths
    return discovered


def load_run_frame(path, run_name):
    df = load_data(path)
    if df.empty:
        print(f"Skipping empty file: {path}")
        return pd.DataFrame()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"Skipping {path}: missing columns {missing}")
        return pd.DataFrame()

    frame = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame["run"] = run_name
    frame["source_file"] = path.name
    return frame


def load_all_runs(data_root):
    frames = []
    for run_name, paths in discover_files(data_root).items():
        existing = [path for path in paths if path.exists()]
        if not existing:
            print(f"No files found for {run_name}")
            continue

        for path in existing:
            frames.append(load_run_frame(path, run_name))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No plottable axis-correlation data loaded.")

    return pd.concat(frames, ignore_index=True)


def print_correlation_summary(df):
    print("Loaded samples:")
    print(df.groupby("run").size().reindex(RUN_ORDER).dropna().astype(int).to_string())
    print()
    print("Correlations by run:")

    for run_name in RUN_ORDER:
        run_df = df[df["run"] == run_name]
        if run_df.empty:
            continue

        corr_xx = run_df["x_target_abs"].corr(run_df["x_actual_abs"])
        corr_xy = run_df["x_target_abs"].corr(run_df["y_actual_abs"])
        corr_yx = run_df["y_target_abs"].corr(run_df["x_actual_abs"])
        corr_yy = run_df["y_target_abs"].corr(run_df["y_actual_abs"])
        print(
            f"{run_name}: "
            f"tx/ax={corr_xx:.4f}, tx/ay={corr_xy:.4f}, "
            f"ty/ax={corr_yx:.4f}, ty/ay={corr_yy:.4f}"
        )


def add_identity_line(ax, x_values, y_values):
    low = min(x_values.min(), y_values.min())
    high = max(x_values.max(), y_values.max())
    ax.plot([low, high], [low, high], color="black", linewidth=0.8, alpha=0.35, zorder=0)


def plot_overlay(df, output, show):
    plot_specs = [
        ("x_target_abs", "x_actual_abs", "Target X vs Actual X", "Expected: diagonal", True),
        ("x_target_abs", "y_actual_abs", "Target X vs Actual Y", "Expected: cloud", False),
        ("y_target_abs", "x_actual_abs", "Target Y vs Actual X", "Expected: cloud", False),
        ("y_target_abs", "y_actual_abs", "Target Y vs Actual Y", "Expected: diagonal", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Axis diagnosis overlay: run70 / run71 / run72", fontsize=16)

    for ax, (x_col, y_col, title, subtitle, show_identity) in zip(axes.flat, plot_specs):
        if show_identity:
            add_identity_line(ax, df[x_col], df[y_col])

        for run_name in RUN_ORDER:
            run_df = df[df["run"] == run_name]
            if run_df.empty:
                continue
            ax.scatter(
                run_df[x_col],
                run_df[y_col],
                s=7,
                alpha=0.36,
                linewidths=0,
                color=RUN_COLORS[run_name],
            )

        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)

    legend_handles = []
    for run_name in RUN_ORDER:
        run_count = len(df[df["run"] == run_name])
        if run_count == 0:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=RUN_COLORS[run_name],
                label=f"{run_name} ({run_count} samples)",
                markersize=7,
                alpha=0.8,
            )
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved plot: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay the four target/actual axis-correlation scatter plots for run70, run71, and run72."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Data root directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_all_runs(args.data_root)
    print_correlation_summary(df)
    plot_overlay(df, args.output, args.show)


if __name__ == "__main__":
    main()
