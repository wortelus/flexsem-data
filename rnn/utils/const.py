import torch

from rnn.models.model_gru import HysteresisGRU
from rnn.models.model_transformer import HysteresisTransformer

# Seed for reproducibility (numpy, torch, etc.)
SEED = 10

# Data splits
TRAIN_SPLIT = 0.7

# We set to 0.0 because WE DON'T WANT TO FILTER ANYTHING, RUN 55-72 ARE CHECKED / FILTERED ALREADY
MIN_CONFIDENCE = 0.0

# Model selection
# MODEL = HysteresisLSTM
MODEL = HysteresisGRU
# MODEL = HysteresisTransformer

INVERSE_MODEL = False

# LSTM/GRU parameters
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 32
NUM_LAYERS = 1
BIDIRECTIONAL = False
INPUT_SIZE = 4  # fixed
OUTPUT_SIZE = 2  # fixed

# transformer specific
N_HEADS = 8

# Loss function
CRITERION = torch.nn.MSELoss()

# Training parameters
OPTIMIZER = torch.optim.Adam
EPOCHS = 1000
BATCH_SIZE = 32

# Scheduler parameters
LEARNING_RATE = 0.0005
SCHEDULER_PATIENCE = 50
SCHEDULER_FACTOR = 0.5
SCHEDULER_THRESHOLD = 1e-6
SCHEDULER_MIN_LR = 1e-7

DROPOUT = 0.1

#
# Directories and file paths
#

# Raw data source
EXPERIMENT_DIR = "data_original"

# Split & processed dataset directory
DATASET_DIR = 'temp/dataset/'
DATASET_POSTFIX = f'_{"inverse" if INVERSE_MODEL else "forward"}_dataset.pt'

# Model output paths
_model = 'transformer' \
    if MODEL == HysteresisTransformer \
    else 'gru' if MODEL == HysteresisGRU \
    else 'lstm'
_prefix = f"{"inverse " if INVERSE_MODEL else "forward"}_{_model}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_b{int(BIDIRECTIONAL)}_seq{SEQUENCE_LENGTH}_nh{N_HEADS if _model == 'transformer' else 0}"

SCALER_PATH = 'temp/scaler_relative.gz'
MODEL_SAVE_PATH = f'temp/{_prefix}_model.pt'
