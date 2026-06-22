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

    def last_position(self) -> tuple[Distance, Distance] | None:
        if len(self.history) == 0:
            return None

        last_entry = self.history[-1]
        return (Distance.from_nanometers(last_entry[2]),
                Distance.from_nanometers(last_entry[3]))  # actual_x, actual_y

    def _update_history(self, target_x: float, target_y: float, actual_x: float, actual_y: float):
        logger.info(
            f"Updating hysteresis history with target=({target_x}, {target_y}), actual=({actual_x}, {actual_y})")
        self.history.append([target_x, target_y, actual_x, actual_y])

    def _prepare_input(self, test_cmd_x, test_cmd_y):
        if len(self.history) < self.window_size:
            raise ValueError("History buffer not full. Call initialize_history() first.")

        # History to numpy
        history_arr = np.array(self.history)  # (seq_len, 4)

        # History with shifted actual command values (3rd and 4th column)
        # That is how the input matrix is constructed for the model
        window = np.empty((self.window_size, self.input_size))  # (seq_len, 4)
        for i in range(1, self.window_size):
            window[i - 1, 0:2] = history_arr[i, 0:2]  # target_x, target_y from step i
            window[i - 1, 2:4] = history_arr[i - 1, 2:4]  # actual_x, actual_y from step i - 1

        # Last row
        window[-1, 0:2] = [test_cmd_x, test_cmd_y]  # target_x, target_y from current test command
        window[-1, 2:4] = history_arr[-1, 2:4]  # actual_x, actual_y from last step

        # Relativize: subtract the first target position from all target and actual positions in the window
        ref_vals = window[0, 0:2]  # target_x, target_y from first step in the window
        ref_full = np.hstack((ref_vals, ref_vals))
        window_rel = window - ref_full  # (seq_len, 4)

        # Scaler: apply forward scaling
        shape_orig = window_rel.shape
        window_scaled = self.scaler.transform(window_rel.reshape(-1, 1)).reshape(shape_orig)  # (seq_len, 4)

        # ONNX input as float32 and (Batch, Seq, Features) -> (1, seq_len, 4)
        # return ref_vals so we can compare the results in predict_actual_position in absolute coords
        return window_scaled.astype(np.float32)[np.newaxis, ...], ref_vals

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

    def update_history(self,
                       x_task: Distance,
                       y_task: Distance,
                       ref_x: Distance,
                       ref_y: Distance,
                       dx_h: Distance,
                       dy_h: Distance,
                       conf_h: float):
        logger.info(f"Updating hysteresis movement with (dx={dx_h}, yy={dy_h}) with conf={conf_h}")

        # Update hysteresis manager history
        self._update_history(x_task.nanometers,
                             y_task.nanometers,
                             ref_x.nanometers - dx_h.nanometers,
                             ref_y.nanometers - dy_h.nanometers)

    # @TODO: revise or remove
    def find_optimal_command(self, target_x: Distance, target_y: Distance) -> tuple[
        tuple[Distance, Distance], Distance]:
        target_x_nm = target_x.nanometers
        target_y_nm = target_y.nanometers

        if len(self.history) < self.window_size:
            logger.error("Hysteresis history not full. Call initialize_history() first.")

        history_arr = np.array(self.history)  # (window_size, 4)

        # Build base window (window_size - 1 rows) with shifted actuals
        base_window = np.empty((self.window_size - 1, self.input_size))
        for i in range(1, self.window_size):
            base_window[i - 1, 0:2] = history_arr[i, 0:2]  # target from step i
            base_window[i - 1, 2:4] = history_arr[i - 1, 2:4]  # actual from step i-1

        last_actual = history_arr[-1, 2:4]

        # Generate candidate grid
        candidates = []
        for x_mul in SEARCH_RANGE:
            for y_mul in SEARCH_RANGE:
                candidates.append([
                    target_x_nm + x_mul * STEP_NM,
                    target_y_nm + y_mul * STEP_NM,
                ])
        candidates = np.array(candidates)  # (N, 2)
        n_candidates = len(candidates)

        # Build last row for each candidate: [cmd_x, cmd_y, last_actual_x, last_actual_y]
        last_rows = np.empty((n_candidates, 1, self.input_size))
        last_rows[:, 0, 0:2] = candidates
        last_rows[:, 0, 2:4] = last_actual

        # Tile base window: (N, window_size-1, 4)
        base_tiled = np.broadcast_to(
            base_window[np.newaxis, :, :],
            (n_candidates, self.window_size - 1, self.input_size)
        ).copy()

        # Full windows: (N, window_size, 4)
        full_windows = np.concatenate([base_tiled, last_rows], axis=1)

        # Relativize
        ref = full_windows[:, 0, 0:2]  # (N, 2)
        ref_full = np.concatenate([ref, ref], axis=1)[:, np.newaxis, :]  # (N, 1, 4)
        windows_rel = full_windows - ref_full

        # Scale
        n, w, f = windows_rel.shape
        windows_scaled = self.scaler.transform(
            windows_rel.reshape(-1, 1)
        ).reshape(n, w, f).astype(np.float32)

        # ===== ONNX INFERENCE =====
        # Single batched ONNX inference
        preds_scaled = self.session.run(
            [self.output_name], {self.input_name: windows_scaled}
        )[0]  # (N, 2)

        # Inverse scale
        preds_rel = self.scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).reshape(-1, 2)

        # Absolutize
        # preds_abs = preds_rel + ref.reshape(-1, 2)  # (N, 2) - note: ref differs per candidate due to relativization...

        # But ref is same for all candidates (base_window[0] is identical)
        # so we can simplify:
        ref_point = full_windows[0, 0, 0:2]
        preds_abs = preds_rel + ref_point

        # Find best
        dists = np.sqrt(np.sum((preds_abs - [target_x_nm, target_y_nm]) ** 2, axis=1))
        best_idx = np.argmin(dists)
        min_error = dists[best_idx]

        best_cmd = (
            Distance.from_nanometers(int(candidates[best_idx, 0])),
            Distance.from_nanometers(int(candidates[best_idx, 1])),
        )

        logger.info(
            f"Hysteresis compensation from "
            f"(x={target_x_nm} nm, y={target_y_nm} nm) to "
            f"(x={best_cmd[0].nanometers} nm, y={best_cmd[1].nanometers} nm) "
            f"with dist={min_error:.1f}nm ({n_candidates} candidates)"
        )
        return best_cmd, Distance.from_nanometers(int(min_error))
