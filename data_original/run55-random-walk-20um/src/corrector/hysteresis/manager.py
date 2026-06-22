# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from collections import deque

import joblib
import numpy as np
import onnxruntime as ort
from flexsem.utils import Distance

from corrector.config.models import HysteresisConfig
from corrector.hysteresis.const import *

logger = logging.getLogger(__name__)


class HysteresisManager:
    def __init__(self, hysteresis_cfg: HysteresisConfig):
        self.enabled = hysteresis_cfg.enabled
        self.model_path = hysteresis_cfg.model_path / MODEL_PATH
        self.scaler_path = hysteresis_cfg.model_path / SCALER_PATH
        self.simulator = hysteresis_cfg.simulator
        self.sanity_limit = hysteresis_cfg.sanity_limit

        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        logger.info(f"Loaded ONNX model from {self.model_path}")

        input_shape = self.session.get_inputs()[0].shape  # e.g. [batch size, sequence length, features (4)]

        self.input_size = input_shape[2]
        self.window_size = input_shape[1]

        logger.info(f"Hysteresis window size: {self.window_size}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"Loaded scaler from {self.scaler_path}")

        self.history = deque(maxlen=self.window_size)

    @property
    def sequence_length(self):
        return self.window_size

    def is_enabled(self) -> bool:
        return self.enabled

    def update_history(self, target_x: float, target_y: float, actual_x: float, actual_y: float):
        logger.info(f"Updating hysteresis history with target=({target_x}, {target_y}), actual=({actual_x}, {actual_y})")
        self.history.append([target_x, target_y, actual_x, actual_y])

    def _prepare_input(self, test_cmd_x, test_cmd_y):
        if len(self.history) < self.window_size:
            raise ValueError("History buffer not full. Call initialize_history() first.")

        # History to numpy
        history_arr = np.array(self.history)  # (seq_len, 4)
        subhistory_arr = history_arr[1:]  # (seq_len - 1, 4)

        # Last step (seq_len) (test command + last actual position as prev_actual)
        last_actual_x = history_arr[-1, 2]
        last_actual_y = history_arr[-1, 3]
        current_step = np.array([[test_cmd_x, test_cmd_y, last_actual_x, last_actual_y]])

        # Window
        raw_window = np.vstack([subhistory_arr, current_step])  # (seq_len, 4)

        # Relativization in respect to the first command in the window
        ref_vals = raw_window[0, 0:2]  # [Tx, Ty]
        ref_full = np.hstack([ref_vals, ref_vals])
        window_rel = raw_window - ref_full

        # Scaling (Flatten -> Transform -> Reshape)
        shape_orig = window_rel.shape
        window_flat = window_rel.reshape(-1, 1)
        window_scaled = self.scaler.transform(window_flat)
        window_final = window_scaled.reshape(shape_orig)

        # ONNX input as float32 and (Batch, Seq, Features) -> (1, seq_len, 4)
        # return ref_vals so we can compare the results in predict_actual_position in absolute coords
        return window_final.astype(np.float32)[np.newaxis, ...], ref_vals

    def _predict_actual_position(self, cmd_x, cmd_y):
        input_tensor, ref_vals = self._prepare_input(cmd_x, cmd_y)

        # === ONNX INFERENCE ===
        # run(output_names, input_feed)
        result = self.session.run([self.output_name], {self.input_name: input_tensor})
        pred_rel_scaled = result[0]  # (1, 2)

        # Inverse scaling
        pred_flat = pred_rel_scaled.reshape(-1, 1)
        pred_rel = self.scaler.inverse_transform(pred_flat).reshape(1, 2)

        # Absolutization
        pred_abs_x = pred_rel[0, 0] + ref_vals[0]
        pred_abs_y = pred_rel[0, 1] + ref_vals[1]

        return pred_abs_x, pred_abs_y

    def find_optimal_command(self, target_x: Distance, target_y: Distance) -> tuple[
        tuple[Distance, Distance], Distance]:
        best_cmd = (target_x, target_y)
        min_error = float('inf')

        target_x_nm = target_x.nanometers
        target_y_nm = target_y.nanometers

        for x_mul in SEARCH_RANGE:
            for y_mul in SEARCH_RANGE:
                test_x_nm = target_x_nm + (x_mul * STEP_NM)
                test_y_nm = target_y_nm + (y_mul * STEP_NM)

                pred_x, pred_y = self._predict_actual_position(test_x_nm, test_y_nm)

                dist = np.sqrt((pred_x - target_x_nm) ** 2 + (pred_y - target_y_nm) ** 2)

                if dist < min_error:
                    min_error = dist
                    best_cmd = (
                        Distance.from_nanometers(test_x_nm),
                        Distance.from_nanometers(test_y_nm)
                    )

        # TODO: better logging
        logger.info(f"Hysteresis compensation from "
                    f"(x={target_x} nm, y={target_y_nm} nm) to "
                    f"(x={best_cmd[0].nanometers} nm, y={best_cmd[1].nanometers} nm) with dist={min_error}nm")
        return best_cmd, Distance.from_nanometers(min_error)
