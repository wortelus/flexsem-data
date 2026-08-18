"""Click-and-run grid training for inverse GRU models.

Edit the CONFIGURATION section below and run this file. Every experiment is
stored directly under ``rnn/outputs/<run_name>/`` using the same basic layout
as ``run_2_fit.py``.

Important: window coordinates and prediction targets are preprocessing
choices, not GRU hyperparameters. A ``delta/residual_delta`` tensor dataset is
not also a ``relative`` dataset. To compare representations, generate one
train/val/test/scaler set per representation and add each set to
``DATASET_VARIANTS``. All GRU configurations within a variant reuse and copy
the exact same source files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

RNN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RNN_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rnn.models.model_gru import HysteresisGRU
from rnn.utils import const as current_config
from rnn.utils.loss import RelativeMSELoss


OUTPUT_ROOT = RNN_DIR / "outputs"


@dataclass(frozen=True)
class DatasetVariant:
    """One already-generated tensor representation shared by many GRU runs."""

    name: str
    train_path: Path
    val_path: Path
    test_path: Path
    scaler_path: Path
    window_coord_mode: str
    target_mode: str
    sequence_length: int
    input_size: int = 4
    output_size: int = 2


@dataclass(frozen=True)
class Experiment:
    dataset: DatasetVariant
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    batch_size: int
    learning_rate: float
    loss_mode: str
    seed: int


# =============================================================================
# CONFIGURATION -- normally this is the only section you need to edit
# =============================================================================

# Every enabled representation needs its own generated .pt files and scaler.
# The default points to the dataset selected by rnn/utils/const.py.
DATASET_VARIANTS = (
    DatasetVariant(
        name="current",
        train_path=current_config.DATASET_DIR_PATH / "train.pt",
        val_path=current_config.DATASET_DIR_PATH / "val.pt",
        test_path=current_config.DATASET_DIR_PATH / "test.pt",
        scaler_path=Path(current_config.SCALER_PATH),
        window_coord_mode=current_config.WINDOW_COORD_MODE,
        target_mode=current_config.TARGET_MODE,
        sequence_length=current_config.SEQUENCE_LENGTH,
        input_size=current_config.INPUT_SIZE,
        output_size=current_config.OUTPUT_SIZE,
    ),

    # To compare all valid representations, generate their datasets first and
    # add entries like these (with real paths):
    # DatasetVariant(
    #     name="delta_actual",
    #     train_path=Path(r"C:\path\delta_actual\train.pt"),
    #     val_path=Path(r"C:\path\delta_actual\val.pt"),
    #     test_path=Path(r"C:\path\delta_actual\test.pt"),
    #     scaler_path=Path(r"C:\path\delta_actual\scaler.gz"),
    #     window_coord_mode="delta",
    #     target_mode="actual_delta",
    #     sequence_length=16,
    # ),
    # DatasetVariant(
    #     name="relative",
    #     train_path=Path(r"C:\path\relative\train.pt"),
    #     val_path=Path(r"C:\path\relative\val.pt"),
    #     test_path=Path(r"C:\path\relative\test.pt"),
    #     scaler_path=Path(r"C:\path\relative\scaler.gz"),
    #     window_coord_mode="relative",
    #     target_mode="actual_delta",  # relative-position target in this mode
    #     sequence_length=16,
    # ),
)

# Cartesian GRU grid. Tuples with one value keep that option fixed.
HIDDEN_SIZES = (32, 64, 128)
NUM_LAYERS = (1, 2)
BIDIRECTIONAL_OPTIONS = (False,)
DROPOUTS = (0.0, 0.1)
BATCH_SIZES = (16, 32)
LEARNING_RATES = (1e-3, 5e-4)
LOSS_MODES = ("mse",)
SEEDS = (10,)

# Shared fit settings.
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 0.0
SCHEDULER_PATIENCE = 50
SCHEDULER_FACTOR = 0.5
SCHEDULER_THRESHOLD = 1e-4
SCHEDULER_MIN_LR = 1e-7
RELATIVE_LOSS_EPS_NM = 10_000.0

# Runtime/output behavior.
RUN_NAME_PREFIX = "grid_"
GRID_SUMMARY_FILENAME = "grid_fit_summary.json"
COMPILE_MODEL = True
SKIP_COMPLETED = True
SKIP_REDUNDANT_SINGLE_LAYER_DROPOUT = True
COPY_SPLIT_MANIFEST = True
NUM_WORKERS = 0


# =============================================================================
# Implementation
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets and print planned experiments without training.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only the first N configurations (useful for a smoke test).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override EPOCHS for this invocation.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for this invocation.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def path_token(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    ).strip("_")
    if not result:
        raise ValueError(f"Dataset variant name has no usable characters: {value!r}")
    return result


def experiment_run_name(experiment: Experiment) -> str:
    loss_tag = (
        "mse"
        if experiment.loss_mode == "mse"
        else f"relative_mse_eps{path_token(int(RELATIVE_LOSS_EPS_NM))}"
    )
    dataset = experiment.dataset
    return (
        f"{RUN_NAME_PREFIX}inverse_gru"
        f"_h{experiment.hidden_size}_l{experiment.num_layers}"
        f"_b{int(experiment.bidirectional)}"
        f"_seq{dataset.sequence_length}"
        f"_{dataset.window_coord_mode}_{dataset.target_mode}"
        f"_nh0_{loss_tag}"
        f"_bs{experiment.batch_size}"
        f"_lr{path_token(experiment.learning_rate)}"
        f"_do{path_token(experiment.dropout)}"
        f"_seed{experiment.seed}"
        f"_ds{safe_name(dataset.name)}"
    )


def build_experiments() -> list[Experiment]:
    experiments = []
    hyperparameter_product = itertools.product(
        HIDDEN_SIZES,
        NUM_LAYERS,
        BIDIRECTIONAL_OPTIONS,
        DROPOUTS,
        BATCH_SIZES,
        LEARNING_RATES,
        LOSS_MODES,
        SEEDS,
    )
    hyperparameters = list(hyperparameter_product)

    for dataset in DATASET_VARIANTS:
        for (
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
            batch_size,
            learning_rate,
            loss_mode,
            seed,
        ) in hyperparameters:
            if (
                SKIP_REDUNDANT_SINGLE_LAYER_DROPOUT
                and num_layers == 1
                and dropout != 0.0
            ):
                continue
            experiments.append(
                Experiment(
                    dataset=dataset,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    dropout=dropout,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    loss_mode=loss_mode,
                    seed=seed,
                )
            )
    return experiments


def validate_grid_configuration(experiments: list[Experiment]) -> None:
    if not DATASET_VARIANTS:
        raise ValueError("DATASET_VARIANTS is empty")
    if not experiments:
        raise ValueError("The configured grid contains no experiments")

    names = [variant.name for variant in DATASET_VARIANTS]
    if len(names) != len(set(names)):
        raise ValueError(f"Dataset variant names must be unique: {names}")

    representations_by_files: dict[tuple[Path, Path, Path], set[tuple[str, str]]] = {}
    for variant in DATASET_VARIANTS:
        if variant.window_coord_mode not in ("delta", "relative"):
            raise ValueError(
                f"{variant.name}: unsupported window_coord_mode "
                f"{variant.window_coord_mode!r}"
            )
        if variant.target_mode not in ("actual_delta", "residual_delta"):
            raise ValueError(
                f"{variant.name}: unsupported target_mode {variant.target_mode!r}"
            )
        if (
            variant.target_mode == "residual_delta"
            and variant.window_coord_mode != "delta"
        ):
            raise ValueError(
                f"{variant.name}: residual_delta requires delta coordinates"
            )

        file_key = tuple(
            resolve_path(path)
            for path in (variant.train_path, variant.val_path, variant.test_path)
        )
        representations_by_files.setdefault(file_key, set()).add(
            (variant.window_coord_mode, variant.target_mode)
        )

    conflicting = {
        files: representations
        for files, representations in representations_by_files.items()
        if len(representations) > 1
    }
    if conflicting:
        raise ValueError(
            "The same .pt files are labelled as multiple representations. "
            "Generate separate datasets for delta/residual_delta, "
            "delta/actual_delta and relative before comparing them. "
            f"Conflicts: {conflicting}"
        )

    valid_loss_modes = {"mse", "relative_mse"}
    unknown_loss_modes = set(LOSS_MODES) - valid_loss_modes
    if unknown_loss_modes:
        raise ValueError(f"Unsupported LOSS_MODES: {sorted(unknown_loss_modes)}")

    if any(value <= 0 for value in HIDDEN_SIZES):
        raise ValueError("HIDDEN_SIZES must be positive")
    if any(value <= 0 for value in NUM_LAYERS):
        raise ValueError("NUM_LAYERS must be positive")
    if any(value <= 0 for value in BATCH_SIZES):
        raise ValueError("BATCH_SIZES must be positive")
    if any(value <= 0 for value in LEARNING_RATES):
        raise ValueError("LEARNING_RATES must be positive")
    if any(not 0.0 <= value < 1.0 for value in DROPOUTS):
        raise ValueError("DROPOUTS must be in [0, 1)")

    run_names = [experiment_run_name(experiment) for experiment in experiments]
    if len(run_names) != len(set(run_names)):
        raise ValueError("The grid produces duplicate run directory names")


def source_config_path(variant: DatasetVariant) -> Path:
    train_path = resolve_path(variant.train_path)
    return train_path.parent.parent / "config.json"


def validate_source_metadata(variant: DatasetVariant) -> None:
    config_path = source_config_path(variant)
    if not config_path.exists():
        print(
            f"  Note: {variant.name}: no source config.json at {config_path}; "
            "representation labels cannot be cross-checked."
        )
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "inverse_model": True,
        "window_coord_mode": variant.window_coord_mode,
        "target_mode": variant.target_mode,
        "sequence_length": variant.sequence_length,
        "input_size": variant.input_size,
        "output_size": variant.output_size,
    }
    mismatches = {
        key: {"source_config": config.get(key), "grid_variant": value}
        for key, value in expected.items()
        if key in config and config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{variant.name}: dataset metadata does not match the grid: "
            f"{mismatches}"
        )


def validate_tensor_dataset(dataset, split: str, variant: DatasetVariant) -> None:
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError(f"{variant.name}/{split}: loaded object is not a dataset")
    if len(dataset) == 0:
        raise ValueError(f"{variant.name}/{split}: dataset is empty")

    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError(
            f"{variant.name}/{split}: expected (sequence, label) samples"
        )
    sequence, label = sample
    expected_sequence_shape = (variant.sequence_length, variant.input_size)
    if tuple(sequence.shape) != expected_sequence_shape:
        raise ValueError(
            f"{variant.name}/{split}: sequence shape {tuple(sequence.shape)} != "
            f"{expected_sequence_shape}"
        )
    if tuple(label.shape) != (variant.output_size,):
        raise ValueError(
            f"{variant.name}/{split}: label shape {tuple(label.shape)} != "
            f"{(variant.output_size,)}"
        )


def load_dataset_bundle(variant: DatasetVariant) -> dict[str, object]:
    paths = {
        "train": resolve_path(variant.train_path),
        "val": resolve_path(variant.val_path),
        "test": resolve_path(variant.test_path),
        "scaler": resolve_path(variant.scaler_path),
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{variant.name}: missing source files: "
            + ", ".join(str(path) for path in missing)
        )

    validate_source_metadata(variant)
    datasets = {
        split: torch.load(paths[split], map_location="cpu", weights_only=False)
        for split in ("train", "val", "test")
    }
    for split, dataset in datasets.items():
        validate_tensor_dataset(dataset, split, variant)

    scaler = joblib.load(paths["scaler"])
    if not hasattr(scaler, "inverse_transform"):
        raise TypeError(f"{variant.name}: scaler has no inverse_transform method")

    print(
        f"  Dataset {variant.name}: "
        f"train={len(datasets['train'])}, val={len(datasets['val'])}, "
        f"test={len(datasets['test'])}"
    )
    return {"paths": paths, "datasets": datasets, "scaler": scaler}


def set_determinism(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker_factory(seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def copy_file(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source == destination.resolve():
        return
    shutil.copy2(source, destination)


def prepare_run_directory(
    experiment: Experiment,
    bundle: dict[str, object],
    epochs: int,
) -> dict[str, Path]:
    run_name = experiment_run_name(experiment)
    run_dir = OUTPUT_ROOT / run_name
    paths = {
        "run": run_dir,
        "dataset": run_dir / "dataset",
        "models": run_dir / "models",
        "scalers": run_dir / "scalers",
        "plots": run_dir / "plots",
        "export": run_dir / "export",
        "model_best": run_dir / "models" / "model.pt.best",
        "config": run_dir / "config.json",
        "fit_result": run_dir / "fit_result.json",
        "history_csv": run_dir / "training_history.csv",
        "loss_plot": run_dir / "plots" / "training_loss.png",
    }
    for key in ("dataset", "models", "scalers", "plots", "export"):
        paths[key].mkdir(parents=True, exist_ok=True)

    source_paths = bundle["paths"]
    for split in ("train", "val", "test"):
        copy_file(source_paths[split], paths["dataset"] / f"{split}.pt")
    copy_file(source_paths["scaler"], paths["scalers"] / "scaler.gz")

    if COPY_SPLIT_MANIFEST:
        split_manifest = source_paths["train"].parent / "split_manifest.json"
        if split_manifest.is_file():
            copy_file(split_manifest, paths["dataset"] / "split_manifest.json")

    config = {
        "model": "gru",
        "inverse_model": True,
        "dataset_variant": experiment.dataset.name,
        "window_coord_mode": experiment.dataset.window_coord_mode,
        "target_mode": experiment.dataset.target_mode,
        "sequence_length": experiment.dataset.sequence_length,
        "input_size": experiment.dataset.input_size,
        "output_size": experiment.dataset.output_size,
        "hidden_size": experiment.hidden_size,
        "num_layers": experiment.num_layers,
        "bidirectional": experiment.bidirectional,
        "dropout": experiment.dropout,
        "batch_size": experiment.batch_size,
        "learning_rate": experiment.learning_rate,
        "loss_mode": experiment.loss_mode,
        "relative_loss_eps": RELATIVE_LOSS_EPS_NM,
        "seed": experiment.seed,
        "epochs": epochs,
        "optimizer": "Adam",
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_threshold": SCHEDULER_THRESHOLD,
        "scheduler_min_lr": SCHEDULER_MIN_LR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "source_dataset": {
            key: str(value) for key, value in source_paths.items()
        },
        "copied_dataset": {
            "train": str(paths["dataset"] / "train.pt"),
            "val": str(paths["dataset"] / "val.pt"),
            "test": str(paths["dataset"] / "test.pt"),
            "scaler": str(paths["scalers"] / "scaler.gz"),
        },
    }
    paths["config"].write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def make_criterion(experiment: Experiment, scaler, device: torch.device):
    if experiment.loss_mode == "mse":
        return torch.nn.MSELoss().to(device)
    if experiment.loss_mode == "relative_mse":
        return RelativeMSELoss(
            eps_nm=RELATIVE_LOSS_EPS_NM,
            scaler=scaler,
        ).to(device)
    raise ValueError(f"Unsupported loss mode: {experiment.loss_mode}")


def save_history(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Validation Loss")
    ax.set_title("Training & Validation Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def weighted_average_loss(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(sequences)
            batch_size = len(sequences)
            total_loss += float(criterion(outputs, labels).item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Validation loader yielded no samples")
    return total_loss / total_samples


def train_experiment(
    experiment: Experiment,
    bundle: dict[str, object],
    paths: dict[str, Path],
    epochs: int,
    device: torch.device,
    compile_model: bool,
) -> dict[str, object]:
    generator = set_determinism(experiment.seed)
    datasets = bundle["datasets"]
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker_factory(experiment.seed),
        generator=generator,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=experiment.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    raw_model = HysteresisGRU(
        input_size=experiment.dataset.input_size,
        hidden_size=experiment.hidden_size,
        output_size=experiment.dataset.output_size,
        num_layers=experiment.num_layers,
        dropout=experiment.dropout,
        bidirectional=experiment.bidirectional,
    ).to(device)
    training_model = raw_model
    compiled = False
    if compile_model:
        print("  Compiling model...")
        try:
            training_model = torch.compile(raw_model)
            compiled = True
        except Exception as exc:
            print(f"  torch.compile setup failed; using eager mode: {exc}")

    criterion = make_criterion(experiment, bundle["scaler"], device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=experiment.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        threshold=SCHEDULER_THRESHOLD,
        min_lr=SCHEDULER_MIN_LR,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        training_model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            try:
                outputs = training_model(sequences)
            except Exception:
                if not compiled:
                    raise
                print("  torch.compile failed during execution; falling back to eager mode.")
                training_model = raw_model
                compiled = False
                outputs = training_model(sequences)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = len(sequences)
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size

        if total_train_samples == 0:
            raise ValueError("Training loader yielded no samples")
        train_loss = total_train_loss / total_train_samples
        val_loss = weighted_average_loss(
            training_model,
            val_loader,
            criterion,
            device,
        )

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": new_lr,
            }
        )
        print(
            f"  Epoch [{epoch}/{epochs}] "
            f"train={train_loss:.6g} val={val_loss:.6g} lr={new_lr:.3g}"
        )
        if new_lr != old_lr:
            print(f"  Learning rate reduced from {old_lr:.8g} to {new_lr:.8g}")

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save the plain GRU state dict even when the training call is compiled.
            torch.save(raw_model.state_dict(), paths["model_best"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"  Early stopping at epoch {epoch}; "
                    f"best={best_val_loss:.6g} at epoch {best_epoch}."
                )
                break

    save_history(history, paths["history_csv"])
    save_loss_plot(history, paths["loss_plot"])
    return {
        "status": "completed",
        "run_name": experiment_run_name(experiment),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "requested_epochs": epochs,
        "elapsed_seconds": time.time() - started,
        "compiled": compiled,
        "model_path": str(paths["model_best"]),
    }


def completed_result(
    paths: dict[str, Path],
    experiment: Experiment,
    epochs: int,
) -> dict[str, object] | None:
    if not SKIP_COMPLETED:
        return None
    if not paths["fit_result"].is_file() or not paths["model_best"].is_file():
        return None
    try:
        result = json.loads(paths["fit_result"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") != "completed":
        return None
    if result.get("requested_epochs") != epochs:
        return None

    config_path = paths["run"] / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "dataset_variant": experiment.dataset.name,
        "window_coord_mode": experiment.dataset.window_coord_mode,
        "target_mode": experiment.dataset.target_mode,
        "sequence_length": experiment.dataset.sequence_length,
        "input_size": experiment.dataset.input_size,
        "output_size": experiment.dataset.output_size,
        "hidden_size": experiment.hidden_size,
        "num_layers": experiment.num_layers,
        "bidirectional": experiment.bidirectional,
        "dropout": experiment.dropout,
        "batch_size": experiment.batch_size,
        "learning_rate": experiment.learning_rate,
        "loss_mode": experiment.loss_mode,
        "seed": experiment.seed,
        "epochs": epochs,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        return None
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else EPOCHS
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    experiments = build_experiments()
    validate_grid_configuration(experiments)
    if args.max_runs is not None:
        experiments = experiments[: args.max_runs]

    print(f"Grid contains {len(experiments)} experiment(s).")
    print("Loading and validating dataset variants...")
    bundles = {
        variant.name: load_dataset_bundle(variant)
        for variant in DATASET_VARIANTS
    }

    for index, experiment in enumerate(experiments, 1):
        print(f"  [{index:03d}/{len(experiments):03d}] {experiment_run_name(experiment)}")

    if args.dry_run:
        print("Dry run complete: no output directories were created and no model was trained.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compile_model = COMPILE_MODEL and not args.no_compile
    print(f"Using device: {device}; torch.compile={compile_model}")

    summary = {
        "status": "running",
        "started_at_unix": time.time(),
        "epochs": epochs,
        "device": str(device),
        "compile_model": compile_model,
        "experiments": [],
    }
    summary_path = OUTPUT_ROOT / GRID_SUMMARY_FILENAME
    write_json(summary_path, summary)
    failures = 0

    for index, experiment in enumerate(experiments, 1):
        run_name = experiment_run_name(experiment)
        run_dir = OUTPUT_ROOT / run_name
        candidate_paths = {
            "run": run_dir,
            "model_best": run_dir / "models" / "model.pt.best",
            "fit_result": run_dir / "fit_result.json",
        }
        previous = completed_result(candidate_paths, experiment, epochs)
        if previous is not None:
            print(f"[{index}/{len(experiments)}] SKIP completed: {run_name}")
            record = dict(previous)
            record["grid_action"] = "skipped_existing"
            summary["experiments"].append(record)
            write_json(summary_path, summary)
            continue

        print("\n" + "=" * 100)
        print(f"[{index}/{len(experiments)}] TRAIN {run_name}")
        print("=" * 100)
        paths = None
        try:
            bundle = bundles[experiment.dataset.name]
            paths = prepare_run_directory(experiment, bundle, epochs)
            result = train_experiment(
                experiment,
                bundle,
                paths,
                epochs,
                device,
                compile_model,
            )
            write_json(paths["fit_result"], result)
            summary["experiments"].append(result)
            print(
                f"COMPLETED {run_name}: best val={result['best_val_loss']:.6g} "
                f"at epoch {result['best_epoch']}"
            )
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            write_json(summary_path, summary)
            raise
        except Exception as exc:
            failures += 1
            failure = {
                "status": "failed",
                "run_name": run_name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            summary["experiments"].append(failure)
            if paths is not None:
                write_json(paths["fit_result"], failure)
            print(f"FAILED {run_name}: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_json(summary_path, summary)

    summary["status"] = "completed" if failures == 0 else "completed_with_failures"
    summary["finished_at_unix"] = time.time()
    summary["failure_count"] = failures
    write_json(summary_path, summary)
    print(f"\nGrid finished. Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
