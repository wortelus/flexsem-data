import json
from pathlib import Path

import torch

from rnn.models.model_gru import HysteresisGRU
from rnn.models.model_transformer import HysteresisTransformer
from rnn.utils.loss import RelativeMSELoss

# Seed for reproducibility (numpy, torch, etc.)
SEED = 10

# Data splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

_split_total = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
if any(split < 0 for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)):
    raise ValueError("TRAIN_SPLIT, VAL_SPLIT and TEST_SPLIT must be non-negative")
if abs(_split_total - 1.0) > 1e-9:
    raise ValueError(
        "TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0, "
        f"got {_split_total}"
    )

# Model selection
# MODEL = HysteresisLSTM
MODEL = HysteresisGRU
# MODEL = HysteresisTransformer

INVERSE_MODEL = True

# Window coordinate representation:
# - "relative": positions are anchored to the first point in each window
# - "delta": first row is zero, following rows are per-step deltas
WINDOW_COORD_MODE = "delta"
if WINDOW_COORD_MODE not in ("relative", "delta"):
    raise ValueError(f"Unsupported WINDOW_COORD_MODE: {WINDOW_COORD_MODE}")

# Prediction target:
# - "actual_delta": predict the direct output delta
#   - forward: actual_delta
#   - inverse: command_delta
# - "residual_delta": predict the residual over the input delta, only for delta models
#   - forward: actual_delta - command_delta
#   - inverse: command_delta - desired_delta
TARGET_MODE = "residual_delta"
if TARGET_MODE not in ("actual_delta", "residual_delta"):
    raise ValueError(f"Unsupported TARGET_MODE: {TARGET_MODE}")
if TARGET_MODE == "residual_delta" and WINDOW_COORD_MODE != "delta":
    raise ValueError("TARGET_MODE='residual_delta' is only supported for delta models")

# LSTM/GRU parameters
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 32
NUM_LAYERS = 2
BIDIRECTIONAL = False
INPUT_SIZE = 4  # fixed
OUTPUT_SIZE = 2  # fixed

# transformer specific
N_HEADS = 8

# Loss function
# - "mse": standard absolute-error MSE in the scaled target space
# - "relative_mse": MSE weighted by target vector magnitude in nm
LOSS_MODE = "mse"
RELATIVE_LOSS_EPS = 10000.0


def make_criterion(scaler=None):
    if LOSS_MODE == "mse":
        return torch.nn.MSELoss()
    if LOSS_MODE == "relative_mse":
        if scaler is None:
            raise ValueError("relative_mse requires a fitted scaler")
        return RelativeMSELoss(eps_nm=RELATIVE_LOSS_EPS, scaler=scaler)
    raise ValueError(f"Unsupported LOSS_MODE: {LOSS_MODE}")


# Training parameters
OPTIMIZER = torch.optim.Adam
EPOCHS = 750
BATCH_SIZE = 16

# Scheduler parameters
LEARNING_RATE = 0.0005
SCHEDULER_PATIENCE = 50
SCHEDULER_FACTOR = 0.5
SCHEDULER_THRESHOLD = 1e-4
SCHEDULER_MIN_LR = 1e-7

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 0.0

DROPOUT = 0.1

#
# Directories and file paths
#

RNN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RNN_DIR.parent

# Raw data source
EXPERIMENT_DIR = "data_original"

_model = 'transformer' \
    if MODEL == HysteresisTransformer \
    else 'gru' if MODEL == HysteresisGRU \
    else 'lstm'
_direction = "inverse" if INVERSE_MODEL else "forward"
_heads = N_HEADS if _model == "transformer" else 0


def _path_token(value):
    return str(value).replace("-", "m").replace(".", "p")


_loss_tag = "mse" if LOSS_MODE == "mse" else f"relative_mse_eps{_path_token(int(RELATIVE_LOSS_EPS))}"

RUN_NAME = (
    f"{_direction}_{_model}"
    f"_h{HIDDEN_SIZE}_l{NUM_LAYERS}_b{int(BIDIRECTIONAL)}"
    f"_seq{SEQUENCE_LENGTH}_{WINDOW_COORD_MODE}_{TARGET_MODE}"
    f"_nh{_heads}_{_loss_tag}"
    f"_bs{BATCH_SIZE}_lr{_path_token(LEARNING_RATE)}"
    f"_do{_path_token(DROPOUT)}_seed{SEED}"
)

OUTPUT_ROOT = RNN_DIR / "outputs"
RUN_DIR = OUTPUT_ROOT / RUN_NAME
DATASET_DIR_PATH = RUN_DIR / "dataset"
MODEL_DIR = RUN_DIR / "models"
SCALER_DIR = RUN_DIR / "scalers"
PLOTS_DIR = RUN_DIR / "plots"
EXPORT_DIR = RUN_DIR / "export"

# Backwards-compatible alias for older scripts that still import TEMP_DIR.
TEMP_DIR = RUN_DIR

# Split & processed dataset paths. DATASET_DIR stays string-compatible with
# existing f"{DATASET_DIR}train{DATASET_POSTFIX}" call sites.
DATASET_DIR = str(DATASET_DIR_PATH) + "/"
DATASET_POSTFIX = ".pt"

SCALER_PATH = str(SCALER_DIR / "scaler.gz")
MODEL_SAVE_PATH = str(MODEL_DIR / "model.pt")
RUN_CONFIG_PATH = RUN_DIR / "config.json"

RUN_CONFIG = {
    "seed": SEED,
    "train_split": TRAIN_SPLIT,
    "val_split": VAL_SPLIT,
    "test_split": TEST_SPLIT,
    "model": _model,
    "inverse_model": INVERSE_MODEL,
    "window_coord_mode": WINDOW_COORD_MODE,
    "target_mode": TARGET_MODE,
    "sequence_length": SEQUENCE_LENGTH,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "bidirectional": BIDIRECTIONAL,
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "n_heads": N_HEADS,
    "loss_mode": LOSS_MODE,
    "relative_loss_eps": RELATIVE_LOSS_EPS,
    "optimizer": OPTIMIZER.__name__,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "scheduler_patience": SCHEDULER_PATIENCE,
    "scheduler_factor": SCHEDULER_FACTOR,
    "scheduler_threshold": SCHEDULER_THRESHOLD,
    "scheduler_min_lr": SCHEDULER_MIN_LR,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    "dropout": DROPOUT,
    "experiment_dir": EXPERIMENT_DIR,
}


def ensure_output_dirs():
    for directory in (DATASET_DIR_PATH, MODEL_DIR, SCALER_DIR, PLOTS_DIR, EXPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    RUN_CONFIG_PATH.write_text(
        json.dumps(RUN_CONFIG, indent=2, sort_keys=True),
        encoding="utf-8",
    )
