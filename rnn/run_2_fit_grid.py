"""Grid-search history length for the best inverse GRU configuration.

Only the sequence/history length changes: 16, 18, 20, 24, 28 and 32. All other
settings are fixed to the best inverse architecture from the preceding grid.
Commands are not quantized during preprocessing.
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
from torch.utils.data import DataLoader, TensorDataset

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
    inverse_model: bool
    window_coord_mode: str
    target_mode: str
    sequence_length: int
    command_quantization_nm: float | None = None
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

GRID_DATASET_ROOT = OUTPUT_ROOT / "_grid_datasets"

# Use raw command metadata; Q50 is intentionally disabled.
COMMAND_QUANTIZATION_OPTIONS = (None,)
COMMON_EVALUATION_QUANTIZATION_NM = None

# Complete preprocessing grid:
# - relative: positions relative to the first point in the window
# - delta/actual_delta: one-step deltas and a direct one-step target
# - delta/residual_delta: one-step deltas and target minus input-step residual
DIRECTION_OPTIONS = (True,)  # inverse only
REPRESENTATION_OPTIONS = (
    ("delta", "residual_delta"),
)
HISTORY_LENGTHS = (6, 8, 10, 12, 14)


def quantization_tag(quantum_nm: float | None) -> str:
    if quantum_nm is None:
        return "qnone"
    return f"q{float(quantum_nm):g}".replace(".", "p")


def representation_tag(window_coord_mode: str, target_mode: str) -> str:
    if window_coord_mode == "relative":
        return "actual_relative"
    target = "residual" if target_mode == "residual_delta" else "actual"
    return f"delta_{target}"


def direction_tag(inverse_model: bool) -> str:
    return "inverse" if inverse_model else "forward"


def generated_dataset_variant(
    inverse_model: bool,
    window_coord_mode: str,
    target_mode: str,
    quantum_nm: float | None,
    sequence_length: int | None = None,
) -> DatasetVariant:
    explicit_sequence_length = sequence_length is not None
    if sequence_length is None:
        sequence_length = current_config.SEQUENCE_LENGTH
    name = (
        f"{direction_tag(inverse_model)}_"
        f"{representation_tag(window_coord_mode, target_mode)}"
        f"_{quantization_tag(quantum_nm)}"
        f"{'_seq' + str(sequence_length) if explicit_sequence_length else ''}"
    )
    root = GRID_DATASET_ROOT / name
    return DatasetVariant(
        name=name,
        train_path=root / "dataset" / "train.pt",
        val_path=root / "dataset" / "val.pt",
        test_path=root / "dataset" / "test.pt",
        scaler_path=root / "scalers" / "scaler.gz",
        inverse_model=inverse_model,
        window_coord_mode=window_coord_mode,
        target_mode=target_mode,
        sequence_length=sequence_length,
        command_quantization_nm=quantum_nm,
        input_size=current_config.INPUT_SIZE,
        output_size=current_config.OUTPUT_SIZE,
    )


DATASET_VARIANTS = tuple(
    generated_dataset_variant(
        inverse_model,
        window_coord_mode,
        target_mode,
        quantum_nm,
        sequence_length,
    )
    for inverse_model in DIRECTION_OPTIONS
    for window_coord_mode, target_mode in REPRESENTATION_OPTIONS
    for quantum_nm in COMMAND_QUANTIZATION_OPTIONS
    for sequence_length in HISTORY_LENGTHS
)


def common_evaluation_variant(
    inverse_model: bool,
    sequence_length: int | None = None,
) -> DatasetVariant:
    candidates = [
        variant
        for variant in DATASET_VARIANTS
        if variant.inverse_model == inverse_model
        and variant.command_quantization_nm
        == COMMON_EVALUATION_QUANTIZATION_NM
        and (
            sequence_length is None
            or variant.sequence_length == sequence_length
        )
    ]
    direct_delta_candidates = [
        variant
        for variant in candidates
        if variant.window_coord_mode == "delta"
        and variant.target_mode == "actual_delta"
    ]
    if len(direct_delta_candidates) == 1:
        return direct_delta_candidates[0]
    # A single residual dataset is also a valid canonical reference: its
    # label plus the final desired/input delta reconstructs the true output
    # delta. This is useful for one-representation ablation grids.
    if not direct_delta_candidates and len(candidates) == 1:
        return candidates[0]
    if len(direct_delta_candidates) != 1:
        raise ValueError(
            "Each direction/sequence length requires exactly one usable "
            f"evaluation dataset; {direction_tag(inverse_model)} "
            f"sequence_length={sequence_length} has "
            f"{[variant.name for variant in candidates]}"
        )
    return direct_delta_candidates[0]

# Best inverse architecture; history length is the only grid axis.
HIDDEN_SIZES = (32,)
NUM_LAYERS = (2,)
BIDIRECTIONAL_OPTIONS = (False,)
DROPOUTS = (0.0,)
BATCH_SIZES = (32,)
LEARNING_RATES = (5e-4,)
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
GRID_RANKING_FILENAME = "grid_fit_ranking.csv"
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
        "--prepare-datasets-only",
        action="store_true",
        help="Generate and validate quantization datasets, then exit.",
    )
    parser.add_argument(
        "--rebuild-datasets",
        action="store_true",
        help="Regenerate quantization datasets even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned datasets and experiments without writing or training.",
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
    q_tag = quantization_tag(dataset.command_quantization_nm)
    return (
        f"{RUN_NAME_PREFIX}{direction_tag(dataset.inverse_model)}_gru"
        f"_h{experiment.hidden_size}_l{experiment.num_layers}"
        f"_b{int(experiment.bidirectional)}"
        f"_seq{dataset.sequence_length}"
        f"_{dataset.window_coord_mode}_{dataset.target_mode}"
        f"_{q_tag}"
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
        # Keep all preprocessing variants of one architecture adjacent so an
        # interrupted grid still contains directly comparable groups.
        for dataset in DATASET_VARIANTS:
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
    quantization_values = {
        variant.command_quantization_nm for variant in DATASET_VARIANTS
    }
    if COMMON_EVALUATION_QUANTIZATION_NM not in quantization_values:
        raise ValueError(
            "COMMON_EVALUATION_QUANTIZATION_NM must be present in the dataset "
            f"grid; got {COMMON_EVALUATION_QUANTIZATION_NM}"
        )
    direction_sequence_pairs = {
        (variant.inverse_model, variant.sequence_length)
        for variant in DATASET_VARIANTS
    }
    for inverse_model, sequence_length in direction_sequence_pairs:
        common_evaluation_variant(inverse_model, sequence_length)

    representations_by_files: dict[
        tuple[Path, Path, Path],
        set[tuple[bool, str, str, float | None, int]],
    ] = {}
    variant_keys = []
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
        if (
            variant.command_quantization_nm is not None
            and variant.command_quantization_nm <= 0
        ):
            raise ValueError(
                f"{variant.name}: command quantization must be positive or None"
            )

        file_key = tuple(
            resolve_path(path)
            for path in (variant.train_path, variant.val_path, variant.test_path)
        )
        representations_by_files.setdefault(file_key, set()).add(
            (
                variant.inverse_model,
                variant.window_coord_mode,
                variant.target_mode,
                variant.command_quantization_nm,
                variant.sequence_length,
            )
        )
        variant_keys.append(
            (
                variant.inverse_model,
                variant.window_coord_mode,
                variant.target_mode,
                variant.command_quantization_nm,
                variant.sequence_length,
            )
        )

    if len(variant_keys) != len(set(variant_keys)):
        raise ValueError(
            "Direction/representation/quantization/sequence combinations "
            "must be unique"
        )

    conflicting = {
        files: representations
        for files, representations in representations_by_files.items()
        if len(representations) > 1
    }
    if conflicting:
        raise ValueError(
            "The same .pt files are labelled as multiple representations. "
            "Generate separate datasets for each representation and "
            "quantization setting before comparing them. "
            f"Conflicts: {conflicting}"
        )

    valid_loss_modes = {"mse", "relative_mse"}
    unknown_loss_modes = set(LOSS_MODES) - valid_loss_modes
    if unknown_loss_modes:
        raise ValueError(f"Unsupported LOSS_MODES: {sorted(unknown_loss_modes)}")

    if any(value <= 0 for value in HIDDEN_SIZES):
        raise ValueError("HIDDEN_SIZES must be positive")
    if any(variant.sequence_length <= 0 for variant in DATASET_VARIANTS):
        raise ValueError("Dataset sequence lengths must be positive")
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


def dataset_variant_paths(variant: DatasetVariant) -> dict[str, Path]:
    return {
        "train": resolve_path(variant.train_path),
        "val": resolve_path(variant.val_path),
        "test": resolve_path(variant.test_path),
        "scaler": resolve_path(variant.scaler_path),
    }


def generate_dataset_variant(variant: DatasetVariant) -> None:
    """Generate one split-identical preprocessing variant."""
    if variant.input_size != 4:
        raise ValueError("Automatic grid dataset generation requires INPUT_SIZE=4")
    from rnn import run_1_generate_dataset as generator
    from rnn.preprocessing import double as preprocessing

    previous_settings = (
        preprocessing.INVERSE_MODEL,
        preprocessing.WINDOW_COORD_MODE,
        preprocessing.TARGET_MODE,
        preprocessing.COMMAND_QUANTIZATION_NM,
        generator.SEQUENCE_LENGTH,
    )
    preprocessing.INVERSE_MODEL = variant.inverse_model
    preprocessing.WINDOW_COORD_MODE = variant.window_coord_mode
    preprocessing.TARGET_MODE = variant.target_mode
    preprocessing.COMMAND_QUANTIZATION_NM = variant.command_quantization_nm
    generator.SEQUENCE_LENGTH = variant.sequence_length

    try:
        print(
            f"Generating {variant.name}: "
            f"direction={direction_tag(variant.inverse_model)}, "
            f"representation={variant.window_coord_mode}/{variant.target_mode}, "
            f"quantization={variant.command_quantization_nm} nm, "
            f"sequence_length={variant.sequence_length}"
        )
        sources = generator.resolve_experiment_sources(
            str(current_config.REPO_ROOT / current_config.EXPERIMENT_DIR),
            generator.EXPERIMENTS,
        )
        (
            (x_train_u, y_train_u),
            (x_val_u, y_val_u),
            (x_test_u, y_test_u),
            _test_meta,
            manifest_units,
        ) = generator.load_and_split_sources(sources)
        if len(x_train_u) == 0 or len(x_val_u) == 0 or len(x_test_u) == 0:
            raise ValueError(
                f"{variant.name}: empty train/val/test split: "
                f"{len(x_train_u)}/{len(x_val_u)}/{len(x_test_u)}"
            )

        rng = np.random.default_rng(current_config.SEED)
        permutation = rng.permutation(len(x_train_u))
        x_train_u = x_train_u[permutation]
        y_train_u = y_train_u[permutation]

        scaler = preprocessing.fit_scalers(x_train_u, y_train_u)
        unscaled = {
            "train": (x_train_u, y_train_u),
            "val": (x_val_u, y_val_u),
            "test": (x_test_u, y_test_u),
        }
        datasets = {}
        for split, (x_values, y_values) in unscaled.items():
            x_scaled, y_scaled = preprocessing.scale_data(
                x_values, y_values, scaler
            )
            datasets[split] = TensorDataset(
                torch.as_tensor(x_scaled, dtype=torch.float32),
                torch.as_tensor(y_scaled, dtype=torch.float32),
            )

        paths = dataset_variant_paths(variant)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            torch.save(datasets[split], paths[split])
        joblib.dump(scaler, paths["scaler"])

        root = paths["train"].parent.parent
        config = {
            "model": "shared_grid_dataset",
            "inverse_model": variant.inverse_model,
            "window_coord_mode": variant.window_coord_mode,
            "target_mode": variant.target_mode,
            "command_quantization_nm": variant.command_quantization_nm,
            "sequence_length": variant.sequence_length,
            "input_size": variant.input_size,
            "output_size": variant.output_size,
            "seed": current_config.SEED,
            "train_split": current_config.TRAIN_SPLIT,
            "val_split": current_config.VAL_SPLIT,
            "test_split": current_config.TEST_SPLIT,
            "source_experiments": list(generator.EXPERIMENTS),
            "samples": {
                split: len(dataset) for split, dataset in datasets.items()
            },
        }
        (root / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest = {
            "version": 2,
            "seed": current_config.SEED,
            "split_fractions": generator.SPLIT_FRACTIONS,
            "total_windows": {
                split: len(dataset) for split, dataset in datasets.items()
            },
            "split_units": manifest_units,
        }
        (paths["train"].parent / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        (
            preprocessing.INVERSE_MODEL,
            preprocessing.WINDOW_COORD_MODE,
            preprocessing.TARGET_MODE,
            preprocessing.COMMAND_QUANTIZATION_NM,
            generator.SEQUENCE_LENGTH,
        ) = previous_settings


def prepare_dataset_variants(rebuild: bool = False) -> bool:
    regenerated = False
    for variant in DATASET_VARIANTS:
        paths = dataset_variant_paths(variant)
        missing = [path for path in paths.values() if not path.is_file()]
        if rebuild or missing:
            if missing:
                print(
                    f"Dataset {variant.name} is incomplete; missing "
                    + ", ".join(str(path) for path in missing)
                )
            generate_dataset_variant(variant)
            regenerated = True
    return regenerated


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
        "inverse_model": variant.inverse_model,
        "window_coord_mode": variant.window_coord_mode,
        "target_mode": variant.target_mode,
        "command_quantization_nm": variant.command_quantization_nm,
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
    paths = dataset_variant_paths(variant)
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
    return {
        "variant": variant,
        "paths": paths,
        "datasets": datasets,
        "scaler": scaler,
    }

def validate_common_evaluation_alignment(
    reference_variant: DatasetVariant,
    bundles: dict[str, dict[str, object]],
) -> None:
    """Ensure every preprocessing variant refers to the same split samples."""
    reference_bundle = bundles[reference_variant.name]
    reference_manifest_path = (
        reference_bundle["paths"]["train"].parent / "split_manifest.json"
    )
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(
            f"Common evaluation manifest is missing: {reference_manifest_path}"
        )
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )

    for variant in DATASET_VARIANTS:
        if (
            variant.inverse_model != reference_variant.inverse_model
            or variant.sequence_length != reference_variant.sequence_length
        ):
            continue

        if (
            variant.input_size != reference_variant.input_size
            or variant.output_size != reference_variant.output_size
        ):
            raise ValueError(
                "Common evaluation requires matching sequence/input/output "
                f"shapes; {variant.name} differs from {reference_variant.name}."
            )
        bundle = bundles[variant.name]
        manifest_path = bundle["paths"]["train"].parent / "split_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest is missing for {variant.name}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != reference_manifest:
            raise ValueError(
                f"{variant.name} and {reference_variant.name} do not have "
                "identical split manifests; common per-sample evaluation would "
                "not be valid. Regenerate both with --rebuild-datasets."
            )
        for split in ("val", "test"):
            if len(bundle["datasets"][split]) != len(
                reference_bundle["datasets"][split]
            ):
                raise ValueError(
                    f"{variant.name}/{split} has a different sample count from "
                    f"{reference_variant.name}/{split}."
                )


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
        "inverse_model": experiment.dataset.inverse_model,
        "dataset_variant": experiment.dataset.name,
        "window_coord_mode": experiment.dataset.window_coord_mode,
        "target_mode": experiment.dataset.target_mode,
        "command_quantization_nm": experiment.dataset.command_quantization_nm,
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


def inverse_transform_with_global_scaler(
    values: np.ndarray,
    scaler,
) -> np.ndarray:
    """Undo the one-feature global scaler while preserving tensor shape."""
    values = np.asarray(values)
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def canonical_step_delta_nm(
    predictions_nm: np.ndarray,
    sequences_nm: np.ndarray,
    variant: DatasetVariant,
) -> np.ndarray:
    """Convert any representation to its physical one-step output delta.

    For inverse models this is the predicted command delta. For forward models
    it is the predicted actual-motion delta.
    """
    if variant.window_coord_mode == "relative":
        # Relative target and the final previous-output feature share the same
        # window reference. Their difference is the final one-step delta.
        return predictions_nm - sequences_nm[:, -1, 2:4]
    if variant.target_mode == "actual_delta":
        return predictions_nm
    if variant.target_mode == "residual_delta":
        # In delta windows the first feature pair is desired motion (inverse)
        # or command motion (forward). Adding it reconstructs the output delta.
        return predictions_nm + sequences_nm[:, -1, 0:2]
    raise ValueError(
        f"Cannot canonicalize {variant.window_coord_mode}/{variant.target_mode}"
    )


def evaluate_on_common_dataset(
    model,
    variant: DatasetVariant,
    model_input_dataset,
    reference_dataset,
    reference_variant: DatasetVariant,
    reference_scaler,
    model_scaler,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a model on shared reference samples in physical nanometres.

    Inputs come from the model's own aligned dataset. Predictions from relative,
    direct-delta and residual-delta targets are reconstructed into the same
    physical one-step delta. Reference labels are converted to the same
    physical delta representation using the configured evaluation dataset.
    """
    if len(model_input_dataset) != len(reference_dataset):
        raise ValueError(
            "Model and common evaluation datasets contain different sample counts: "
            f"{len(model_input_dataset)} != {len(reference_dataset)}"
        )
    model_loader = DataLoader(
        model_input_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    errors = []
    model.eval()
    with torch.no_grad():
        for (model_sequences, _), (reference_sequences, reference_labels) in zip(
            model_loader, reference_loader, strict=True
        ):
            outputs = model(model_sequences.to(device, non_blocking=True))
            predictions_nm = inverse_transform_with_global_scaler(
                outputs.detach().cpu().numpy(), model_scaler
            )
            sequences_nm = inverse_transform_with_global_scaler(
                model_sequences.numpy(), model_scaler
            )
            labels_nm = inverse_transform_with_global_scaler(
                reference_labels.numpy(), reference_scaler
            )
            reference_sequences_nm = inverse_transform_with_global_scaler(
                reference_sequences.numpy(), reference_scaler
            )
            canonical_predictions_nm = canonical_step_delta_nm(
                predictions_nm,
                sequences_nm,
                variant,
            )
            canonical_labels_nm = canonical_step_delta_nm(
                labels_nm,
                reference_sequences_nm,
                reference_variant,
            )
            errors.append(canonical_predictions_nm - canonical_labels_nm)

    if not errors:
        raise ValueError("Common evaluation dataset yielded no samples")
    errors_nm = np.concatenate(errors, axis=0)
    squared_errors = np.square(errors_nm)
    error_distances = np.linalg.norm(errors_nm, axis=1)
    return {
        "samples": int(len(errors_nm)),
        "mae_nm": float(np.mean(np.abs(errors_nm))),
        "rmse_nm": float(np.sqrt(np.mean(squared_errors))),
        "rmse_x_nm": float(np.sqrt(np.mean(squared_errors[:, 0]))),
        "rmse_y_nm": float(np.sqrt(np.mean(squared_errors[:, 1]))),
        "error_distance_p50_nm": float(np.percentile(error_distances, 50)),
        "error_distance_p90_nm": float(np.percentile(error_distances, 90)),
    }


def train_experiment(
    experiment: Experiment,
    bundle: dict[str, object],
    common_evaluation_bundle: dict[str, object],
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
    raw_model.load_state_dict(
        torch.load(paths["model_best"], map_location=device, weights_only=True)
    )
    common_val_metrics = evaluate_on_common_dataset(
        raw_model,
        experiment.dataset,
        bundle["datasets"]["val"],
        common_evaluation_bundle["datasets"]["val"],
        common_evaluation_bundle["variant"],
        common_evaluation_bundle["scaler"],
        bundle["scaler"],
        experiment.batch_size,
        device,
    )
    common_test_metrics = evaluate_on_common_dataset(
        raw_model,
        experiment.dataset,
        bundle["datasets"]["test"],
        common_evaluation_bundle["datasets"]["test"],
        common_evaluation_bundle["variant"],
        common_evaluation_bundle["scaler"],
        bundle["scaler"],
        experiment.batch_size,
        device,
    )
    return {
        "status": "completed",
        "run_name": experiment_run_name(experiment),
        "dataset_variant": experiment.dataset.name,
        "inverse_model": experiment.dataset.inverse_model,
        "direction": direction_tag(experiment.dataset.inverse_model),
        "command_quantization_nm": experiment.dataset.command_quantization_nm,
        "sequence_length": experiment.dataset.sequence_length,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "requested_epochs": epochs,
        "elapsed_seconds": time.time() - started,
        "compiled": compiled,
        "model_path": str(paths["model_best"]),
        "common_evaluation_quantization_nm": COMMON_EVALUATION_QUANTIZATION_NM,
        "common_evaluation_dataset": common_evaluation_bundle["variant"].name,
        "evaluation_val_metrics": common_val_metrics,
        "evaluation_test_metrics": common_test_metrics,
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
    if result.get("common_evaluation_quantization_nm") != (
        COMMON_EVALUATION_QUANTIZATION_NM
    ):
        return None
    expected_common_dataset = common_evaluation_variant(
        experiment.dataset.inverse_model,
        experiment.dataset.sequence_length,
    ).name
    if result.get("common_evaluation_dataset") != expected_common_dataset:
        return None
    if not isinstance(result.get("evaluation_val_metrics"), dict):
        return None
    if not isinstance(result.get("evaluation_test_metrics"), dict):
        return None

    config_path = paths["run"] / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "dataset_variant": experiment.dataset.name,
        "inverse_model": experiment.dataset.inverse_model,
        "window_coord_mode": experiment.dataset.window_coord_mode,
        "target_mode": experiment.dataset.target_mode,
        "command_quantization_nm": experiment.dataset.command_quantization_nm,
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


def write_ranking_csv(path: Path, records: list[dict[str, object]]) -> None:
    """Write completed runs ranked by validation RMSE in nanometres."""
    completed = [
        record
        for record in records
        if record.get("status") == "completed"
        and isinstance(record.get("evaluation_val_metrics"), dict)
        and isinstance(record.get("evaluation_test_metrics"), dict)
    ]
    completed.sort(
        key=lambda record: record["evaluation_val_metrics"]["rmse_nm"]
    )
    fieldnames = [
        "rank",
        "run_name",
        "dataset_variant",
        "sequence_length",
        "command_quantization_nm",
        "best_epoch",
        "best_scaled_val_loss",
        "val_mae_nm",
        "val_rmse_nm",
        "val_error_distance_p90_nm",
        "test_mae_nm",
        "test_rmse_nm",
        "test_error_distance_p90_nm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(completed, 1):
            val_metrics = record["evaluation_val_metrics"]
            test_metrics = record["evaluation_test_metrics"]
            writer.writerow(
                {
                    "rank": rank,
                    "run_name": record["run_name"],
                    "dataset_variant": record.get("dataset_variant"),
                    "sequence_length": record.get("sequence_length"),
                    "command_quantization_nm": record.get(
                        "command_quantization_nm"
                    ),
                    "best_epoch": record.get("best_epoch"),
                    "best_scaled_val_loss": record.get("best_val_loss"),
                    "val_mae_nm": val_metrics["mae_nm"],
                    "val_rmse_nm": val_metrics["rmse_nm"],
                    "val_error_distance_p90_nm": val_metrics[
                        "error_distance_p90_nm"
                    ],
                    "test_mae_nm": test_metrics["mae_nm"],
                    "test_rmse_nm": test_metrics["rmse_nm"],
                    "test_error_distance_p90_nm": test_metrics[
                        "error_distance_p90_nm"
                    ],
                }
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
    for index, experiment in enumerate(experiments, 1):
        print(f"  [{index:03d}/{len(experiments):03d}] {experiment_run_name(experiment)}")

    if args.dry_run:
        for variant in DATASET_VARIANTS:
            paths = dataset_variant_paths(variant)
            missing = [path for path in paths.values() if not path.is_file()]
            action = "would rebuild" if args.rebuild_datasets else "would generate"
            if missing or args.rebuild_datasets:
                print(f"  Dataset {variant.name}: {action}")
            else:
                load_dataset_bundle(variant)
        print("Dry run complete: nothing was written and no model was trained.")
        return 0

    datasets_regenerated = prepare_dataset_variants(
        rebuild=args.rebuild_datasets
    )
    print("Loading and validating dataset variants...")
    bundles = {
        variant.name: load_dataset_bundle(variant)
        for variant in DATASET_VARIANTS
    }

    direction_sequence_pairs = sorted(
        {
            (variant.inverse_model, variant.sequence_length)
            for variant in DATASET_VARIANTS
        }
    )
    common_evaluation_bundles = {}
    for direction, sequence_length in direction_sequence_pairs:
        ref_variant = common_evaluation_variant(direction, sequence_length)
        validate_common_evaluation_alignment(ref_variant, bundles)
        common_evaluation_bundles[(direction, sequence_length)] = bundles[
            ref_variant.name
        ]

    if args.prepare_datasets_only:
        print("Quantization datasets are ready; no model was trained.")
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
        "common_evaluation_datasets": {
            f"{direction_tag(direction)}_seq{sequence_length}": (
                common_evaluation_variant(direction, sequence_length).name
            )
            for direction, sequence_length in direction_sequence_pairs
        },
        "common_evaluation_quantization_nm": COMMON_EVALUATION_QUANTIZATION_NM,
        "experiments": [],
    }
    summary_path = OUTPUT_ROOT / GRID_SUMMARY_FILENAME
    ranking_path = OUTPUT_ROOT / GRID_RANKING_FILENAME
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
        previous = (
            None
            if datasets_regenerated
            else completed_result(candidate_paths, experiment, epochs)
        )
        if previous is not None:
            print(f"[{index}/{len(experiments)}] SKIP completed: {run_name}")
            record = dict(previous)
            record["grid_action"] = "skipped_existing"
            summary["experiments"].append(record)
            write_json(summary_path, summary)
            write_ranking_csv(ranking_path, summary["experiments"])
            continue

        print("\n" + "=" * 100)
        print(f"[{index}/{len(experiments)}] TRAIN {run_name}")
        print("=" * 100)
        paths = None
        try:
            bundle = bundles[experiment.dataset.name]
            # NOVĚ:
            common_bundle = common_evaluation_bundles[
                (
                    experiment.dataset.inverse_model,
                    experiment.dataset.sequence_length,
                )
            ]
            paths = prepare_run_directory(experiment, bundle, epochs)
            result = train_experiment(
                experiment,
                bundle,
                common_bundle,
                paths,
                epochs,
                device,
                compile_model,
            )
            write_json(paths["fit_result"], result)
            summary["experiments"].append(result)
            print(
                f"COMPLETED {run_name}: best val={result['best_val_loss']:.6g} "
                f"at epoch {result['best_epoch']}; validation "
                f"RMSE={result['evaluation_val_metrics']['rmse_nm']:.2f} nm"
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
            write_ranking_csv(ranking_path, summary["experiments"])

    summary["status"] = "completed" if failures == 0 else "completed_with_failures"
    summary["finished_at_unix"] = time.time()
    summary["failure_count"] = failures
    write_json(summary_path, summary)
    write_ranking_csv(ranking_path, summary["experiments"])
    print(f"\nGrid finished. Summary: {summary_path}")
    print(f"Validation ranking: {ranking_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
