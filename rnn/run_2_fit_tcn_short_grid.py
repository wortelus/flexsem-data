"""Run a small hyperparameter grid around the currently configured TCN.

The script reuses the train/validation tensors and scaler produced by
``python -m rnn.run_1_generate_dataset`` for the current configuration.  It
does not regenerate or copy the dataset for every trial.

The defaults intentionally form a small eight-run screening grid.  Edit the
CONFIGURATION section to change it, or use the command-line overrides for a
quick smoke test::

    python -m rnn.run_2_fit_tcn_short_grid --dry-run
    python -m rnn.run_2_fit_tcn_short_grid --epochs 2 --max-runs 1 --no-compile

Results are written below ``rnn/outputs/tcn_short_grid/<dataset-run>/``.
Trials are ranked only by validation command-delta RMSE in physical
nanometres; the test split is deliberately not consulted during
hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

RNN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RNN_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rnn.models.model_tcn import HysteresisTCN
from rnn.run_2_fit_grid import (
    DatasetVariant,
    Experiment,
    evaluate_command_delta_metrics,
    save_history,
    save_loss_plot,
    seed_worker_factory,
    set_determinism,
    weighted_average_loss,
)
from rnn.utils import const as current_config
from rnn.utils.loss import RelativeMSELoss


# =============================================================================
# CONFIGURATION -- this is normally the only section to edit
# =============================================================================

# Eight trials around the current h32/l2/dropout=0.1 TCN.
HIDDEN_SIZES = (32, 64)
NUM_LAYERS = (2, 3)
DROPOUTS = (0.0, 0.1)

# Keep these fixed for a short, interpretable architecture search.
TCN_KERNEL_SIZES = (current_config.TCN_KERNEL_SIZE,)
BATCH_SIZES = (32,)
LEARNING_RATES = (current_config.LEARNING_RATE,)
SEEDS = (current_config.SEED,)

# A screening run is shorter than the final fit.  Override with --epochs.
EPOCHS = min(current_config.EPOCHS, 250)
EARLY_STOPPING_PATIENCE = min(current_config.EARLY_STOPPING_PATIENCE, 50)
SCHEDULER_PATIENCE = min(current_config.SCHEDULER_PATIENCE, 20)

OUTPUT_ROOT = RNN_DIR / "outputs" / "tcn_short_grid"
COMPILE_MODEL = True
NUM_WORKERS = 0
SKIP_COMPLETED = True


@dataclass(frozen=True)
class GridPoint:
    hidden_size: int
    num_layers: int
    kernel_size: int
    dropout: float
    batch_size: int
    learning_rate: float
    seed: int

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2**self.num_layers - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-run",
        type=Path,
        default=current_config.RUN_DIR,
        help=(
            "Run directory containing dataset/{train,val}.pt and "
            "scalers/scaler.gz (default: the current const.py RUN_DIR)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum epochs per trial (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only the first N grid points, useful for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid and dataset paths without writing or training.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Train completed matching trials again.",
    )
    return parser.parse_args()


def path_token(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def safe_name(value: str) -> str:
    name = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    ).strip("_")
    return name or "dataset"


def build_grid() -> list[GridPoint]:
    points = [
        GridPoint(*values)
        for values in itertools.product(
            HIDDEN_SIZES,
            NUM_LAYERS,
            TCN_KERNEL_SIZES,
            DROPOUTS,
            BATCH_SIZES,
            LEARNING_RATES,
            SEEDS,
        )
    ]
    current = GridPoint(
        hidden_size=current_config.HIDDEN_SIZE,
        num_layers=current_config.NUM_LAYERS,
        kernel_size=current_config.TCN_KERNEL_SIZE,
        dropout=current_config.DROPOUT,
        batch_size=current_config.BATCH_SIZE,
        learning_rate=current_config.LEARNING_RATE,
        seed=current_config.SEED,
    )
    # Put the current model first when it is part of the grid, so --max-runs 1
    # is a useful end-to-end smoke test and creates a baseline result.
    if current in points:
        points.remove(current)
        points.insert(0, current)
    return points


def validate_grid(points: list[GridPoint]) -> None:
    if current_config.MODEL is not HysteresisTCN:
        raise ValueError("The current MODEL in rnn/utils/const.py must be HysteresisTCN")
    if not current_config.INVERSE_MODEL:
        raise ValueError("This short grid currently supports the inverse TCN only")
    if current_config.VAL_SPLIT <= 0:
        raise ValueError("Grid search requires a non-empty validation split")
    if not points:
        raise ValueError("The grid is empty")
    if len(points) != len(set(points)):
        raise ValueError("The grid contains duplicate configurations")
    if any(point.hidden_size <= 0 or point.num_layers <= 0 for point in points):
        raise ValueError("hidden_size and num_layers must be positive")
    if any(point.kernel_size < 2 for point in points):
        raise ValueError("TCN kernel sizes must be at least 2")
    if any(not 0.0 <= point.dropout < 1.0 for point in points):
        raise ValueError("dropout must be in [0, 1)")
    if any(point.batch_size <= 0 or point.learning_rate <= 0 for point in points):
        raise ValueError("batch size and learning rate must be positive")


def run_name(point: GridPoint) -> str:
    return (
        f"tcn_h{point.hidden_size}_l{point.num_layers}_k{point.kernel_size}"
        f"_do{path_token(point.dropout)}_bs{point.batch_size}"
        f"_lr{path_token(point.learning_rate)}_seed{point.seed}"
    )


def resolve_dataset_run(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def source_paths(dataset_run: Path) -> dict[str, Path]:
    return {
        "train": dataset_run / "dataset" / "train.pt",
        "val": dataset_run / "dataset" / "val.pt",
        "test": dataset_run / "dataset" / "test.pt",
        "scaler": dataset_run / "scalers" / "scaler.gz",
        "config": dataset_run / "config.json",
    }


def print_plan(points: list[GridPoint], dataset_run: Path) -> None:
    paths = source_paths(dataset_run)
    print(f"Dataset run: {dataset_run}")
    for name in ("train", "val", "scaler"):
        status = "OK" if paths[name].is_file() else "MISSING"
        print(f"  {name:>6}: {status:7} {paths[name]}")
    print(f"Grid contains {len(points)} trial(s):")
    for index, point in enumerate(points, 1):
        coverage = (
            "covers"
            if point.receptive_field >= current_config.SEQUENCE_LENGTH
            else "shorter than"
        )
        print(
            f"  [{index:02d}/{len(points):02d}] {run_name(point)} "
            f"(receptive field {point.receptive_field}, {coverage} "
            f"seq={current_config.SEQUENCE_LENGTH})"
        )


def validate_source_config(config_path: Path) -> None:
    """Reject a same-shaped dataset with incompatible preprocessing metadata."""
    if not config_path.is_file():
        print(f"Warning: no source config found at {config_path}; checking shapes only.")
        return
    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source dataset config {config_path}: {exc}") from exc

    expected = {
        "inverse_model": current_config.INVERSE_MODEL,
        "window_coord_mode": current_config.WINDOW_COORD_MODE,
        "target_mode": current_config.TARGET_MODE,
        "command_quantization_nm": current_config.COMMAND_QUANTIZATION_NM,
        "sequence_length": current_config.SEQUENCE_LENGTH,
        "input_size": current_config.INPUT_SIZE,
        "output_size": current_config.OUTPUT_SIZE,
    }
    mismatches = {
        name: {"source": source_config[name], "current": value}
        for name, value in expected.items()
        if name in source_config and source_config[name] != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}: source={values['source']!r}, current={values['current']!r}"
            for name, values in mismatches.items()
        )
        raise ValueError(f"Source dataset is incompatible with current const.py ({details})")


def load_source_bundle(dataset_run: Path) -> dict[str, object]:
    paths = source_paths(dataset_run)
    missing = [paths[name] for name in ("train", "val", "scaler") if not paths[name].is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing grid-search input files:\n  "
            f"{formatted}\nGenerate the current dataset first with "
            "`python -m rnn.run_1_generate_dataset`, or pass --dataset-run."
        )

    validate_source_config(paths["config"])
    datasets = {
        "train": torch.load(paths["train"], weights_only=False),
        "val": torch.load(paths["val"], weights_only=False),
    }
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Both training and validation datasets must be non-empty")

    sample_x, sample_y = datasets["train"][0]
    expected_x = (current_config.SEQUENCE_LENGTH, current_config.INPUT_SIZE)
    expected_y = (current_config.OUTPUT_SIZE,)
    if tuple(sample_x.shape) != expected_x or tuple(sample_y.shape) != expected_y:
        raise ValueError(
            "Dataset shape does not match current const.py: "
            f"got X{tuple(sample_x.shape)}, y{tuple(sample_y.shape)}; "
            f"expected X{expected_x}, y{expected_y}"
        )

    return {
        "datasets": datasets,
        "scaler": joblib.load(paths["scaler"]),
        "paths": paths,
    }


def as_grid_experiment(point: GridPoint, dataset_run: Path) -> Experiment:
    paths = source_paths(dataset_run)
    dataset = DatasetVariant(
        name=dataset_run.name,
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        scaler_path=paths["scaler"],
        window_coord_mode=current_config.WINDOW_COORD_MODE,
        target_mode=current_config.TARGET_MODE,
        sequence_length=current_config.SEQUENCE_LENGTH,
        command_quantization_nm=current_config.COMMAND_QUANTIZATION_NM,
        input_size=current_config.INPUT_SIZE,
        output_size=current_config.OUTPUT_SIZE,
    )
    return Experiment(
        dataset=dataset,
        model_type="tcn",
        hidden_size=point.hidden_size,
        num_layers=point.num_layers,
        tcn_kernel_size=point.kernel_size,
        bidirectional=False,
        dropout=point.dropout,
        batch_size=point.batch_size,
        learning_rate=point.learning_rate,
        loss_mode=current_config.LOSS_MODE,
        seed=point.seed,
    )


def grid_output_dir(dataset_run: Path) -> Path:
    return OUTPUT_ROOT / safe_name(dataset_run.name)


def trial_paths(point: GridPoint, dataset_run: Path) -> dict[str, Path]:
    run_dir = grid_output_dir(dataset_run) / run_name(point)
    return {
        "run": run_dir,
        "models": run_dir / "models",
        "plots": run_dir / "plots",
        "model_best": run_dir / "models" / "model.pt.best",
        "config": run_dir / "config.json",
        "result": run_dir / "fit_result.json",
        "history": run_dir / "training_history.csv",
        "plot": run_dir / "plots" / "training_loss.png",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def point_dict(point: GridPoint) -> dict[str, object]:
    return {field.name: getattr(point, field.name) for field in fields(point)}


def prepare_trial(point: GridPoint, dataset_run: Path, epochs: int) -> dict[str, Path]:
    paths = trial_paths(point, dataset_run)
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["plots"].mkdir(parents=True, exist_ok=True)
    write_json(
        paths["config"],
        {
            "model": "tcn",
            **point_dict(point),
            "receptive_field": point.receptive_field,
            "sequence_length": current_config.SEQUENCE_LENGTH,
            "input_size": current_config.INPUT_SIZE,
            "output_size": current_config.OUTPUT_SIZE,
            "inverse_model": current_config.INVERSE_MODEL,
            "window_coord_mode": current_config.WINDOW_COORD_MODE,
            "target_mode": current_config.TARGET_MODE,
            "command_quantization_nm": current_config.COMMAND_QUANTIZATION_NM,
            "loss_mode": current_config.LOSS_MODE,
            "relative_loss_eps": current_config.RELATIVE_LOSS_EPS,
            "epochs": epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "source_dataset_run": str(dataset_run),
            "source_dataset": {
                name: str(path) for name, path in source_paths(dataset_run).items()
            },
        },
    )
    return paths


def make_criterion(scaler, device: torch.device) -> torch.nn.Module:
    if current_config.LOSS_MODE == "mse":
        return torch.nn.MSELoss().to(device)
    if current_config.LOSS_MODE == "relative_mse":
        return RelativeMSELoss(
            eps_nm=current_config.RELATIVE_LOSS_EPS,
            scaler=scaler,
        ).to(device)
    raise ValueError(f"Unsupported LOSS_MODE: {current_config.LOSS_MODE}")


def completed_result(point: GridPoint, dataset_run: Path, epochs: int) -> dict | None:
    paths = trial_paths(point, dataset_run)
    if not SKIP_COMPLETED or not paths["result"].is_file() or not paths["model_best"].is_file():
        return None
    try:
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        **point_dict(point),
        "epochs": epochs,
        "source_dataset_run": str(dataset_run),
    }
    if result.get("status") != "completed":
        return None
    if any(config.get(name) != value for name, value in expected.items()):
        return None
    return result


def train_trial(
    point: GridPoint,
    dataset_run: Path,
    bundle: dict[str, object],
    epochs: int,
    device: torch.device,
    compile_model: bool,
) -> dict[str, object]:
    paths = prepare_trial(point, dataset_run, epochs)
    experiment = as_grid_experiment(point, dataset_run)
    datasets = bundle["datasets"]
    generator = set_determinism(point.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=point.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker_factory(point.seed),
        generator=generator,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=point.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    raw_model = HysteresisTCN(
        input_size=current_config.INPUT_SIZE,
        hidden_size=point.hidden_size,
        output_size=current_config.OUTPUT_SIZE,
        num_layers=point.num_layers,
        dropout=point.dropout,
        tcn_kernel_size=point.kernel_size,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    model = raw_model
    compiled = False
    if compile_model:
        try:
            print("  Compiling model...")
            model = torch.compile(raw_model)
            compiled = True
        except Exception as exc:
            print(f"  torch.compile setup failed; using eager mode: {exc}")

    criterion = make_criterion(bundle["scaler"], device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=point.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=current_config.SCHEDULER_FACTOR,
        threshold=current_config.SCHEDULER_THRESHOLD,
        min_lr=current_config.SCHEDULER_MIN_LR,
    )
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                predictions = model(sequences)
            except Exception:
                if not compiled:
                    raise
                print("  torch.compile failed during execution; falling back to eager mode.")
                model = raw_model
                compiled = False
                predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(sequences)
            total_samples += len(sequences)

        train_loss = total_loss / total_samples
        val_loss = weighted_average_loss(model, val_loader, criterion, device)
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"  Epoch [{epoch}/{epochs}] train={train_loss:.6g} "
            f"val={val_loss:.6g} lr={learning_rate:.3g}"
        )
        if learning_rate != old_lr:
            print(f"  Learning rate reduced from {old_lr:.8g} to {learning_rate:.8g}")

        if val_loss < best_val_loss - current_config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(raw_model.state_dict(), paths["model_best"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    save_history(history, paths["history"])
    save_loss_plot(history, paths["plot"])
    raw_model.load_state_dict(
        torch.load(paths["model_best"], map_location=device, weights_only=True)
    )
    val_metrics = evaluate_command_delta_metrics(
        raw_model,
        datasets["val"],
        experiment,
        bundle["scaler"],
        device,
    )
    return {
        "status": "completed",
        "run_name": run_name(point),
        **point_dict(point),
        "receptive_field": point.receptive_field,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_command_metrics": val_metrics,
        "epochs_completed": len(history),
        "requested_epochs": epochs,
        "elapsed_seconds": time.time() - started,
        "compiled": compiled,
        "model_path": str(paths["model_best"]),
    }


def write_summary_csv(results: list[dict[str, object]], path: Path) -> None:
    rows = []
    for result in results:
        if result.get("status") != "completed":
            continue
        metrics = result["val_command_metrics"]
        rows.append(
            {
                "rank": 0,
                "run_name": result["run_name"],
                "hidden_size": result["hidden_size"],
                "num_layers": result["num_layers"],
                "kernel_size": result["kernel_size"],
                "dropout": result["dropout"],
                "batch_size": result["batch_size"],
                "learning_rate": result["learning_rate"],
                "seed": result["seed"],
                "receptive_field": result["receptive_field"],
                "parameter_count": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                "val_rmse_nm": metrics["rmse_nm"],
                "val_mae_nm": metrics["mae_nm"],
                "val_error_distance_p90_nm": metrics["error_distance_p90_nm"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    rows.sort(key=lambda row: row["val_rmse_nm"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    points = build_grid()
    validate_grid(points)
    if args.max_runs is not None:
        points = points[: args.max_runs]
    dataset_run = resolve_dataset_run(args.dataset_run)
    print_plan(points, dataset_run)
    if args.dry_run:
        print("Dry run complete; nothing was written.")
        return 0

    bundle = load_source_bundle(dataset_run)
    output_dir = grid_output_dir(dataset_run)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile_model = COMPILE_MODEL and not args.no_compile
    print(f"Using device: {device}; torch.compile={compile_model}")
    summary = {
        "status": "running",
        "dataset_run": str(dataset_run),
        "epochs": args.epochs,
        "device": str(device),
        "started_at_unix": time.time(),
        "experiments": [],
    }
    write_json(summary_json, summary)
    failures = 0

    for index, point in enumerate(points, 1):
        previous = None if args.rerun else completed_result(point, dataset_run, args.epochs)
        if previous is not None:
            print(f"[{index}/{len(points)}] SKIP completed: {run_name(point)}")
            summary["experiments"].append(previous)
            write_json(summary_json, summary)
            continue

        print("\n" + "=" * 90)
        print(f"[{index}/{len(points)}] TRAIN {run_name(point)}")
        print("=" * 90)
        try:
            result = train_trial(
                point,
                dataset_run,
                bundle,
                args.epochs,
                device,
                compile_model,
            )
            write_json(trial_paths(point, dataset_run)["result"], result)
            summary["experiments"].append(result)
            print(
                f"COMPLETED: val RMSE={result['val_command_metrics']['rmse_nm']:.2f} nm, "
                f"best epoch={result['best_epoch']}"
            )
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            write_json(summary_json, summary)
            write_summary_csv(summary["experiments"], summary_csv)
            raise
        except Exception as exc:
            failures += 1
            failure = {
                "status": "failed",
                "run_name": run_name(point),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            summary["experiments"].append(failure)
            write_json(trial_paths(point, dataset_run)["result"], failure)
            print(f"FAILED: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_json(summary_json, summary)
            write_summary_csv(summary["experiments"], summary_csv)

    summary["status"] = "completed" if failures == 0 else "completed_with_failures"
    summary["failure_count"] = failures
    summary["finished_at_unix"] = time.time()
    completed = [
        result for result in summary["experiments"] if result.get("status") == "completed"
    ]
    if completed:
        best = min(completed, key=lambda result: result["val_command_metrics"]["rmse_nm"])
        summary["best_run"] = best["run_name"]
        summary["best_val_rmse_nm"] = best["val_command_metrics"]["rmse_nm"]
    write_json(summary_json, summary)
    write_summary_csv(summary["experiments"], summary_csv)
    print(f"\nGrid finished. Ranked results: {summary_csv}")
    if completed:
        print(
            f"Best run: {summary['best_run']} "
            f"({summary['best_val_rmse_nm']:.2f} nm validation RMSE)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
