# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from pathlib import Path

STEP_NM = 100
SEARCH_RANGE = range(-50, 50)

MODEL_PATH = Path("./model.onnx")
SCALER_PATH = Path("./scaler.pkl")
SCALER_FALLBACK_PATH = Path("./scaler.gz")
CONFIG_PATH = Path("./config.json")

# Fallbacks for older model folders without config.json. Keep these in sync
# with the rnn/utils/const.py settings used to export the model.
INVERSE_MODEL = False
WINDOW_COORD_MODE = "relative"
TARGET_MODE = "actual_delta"
