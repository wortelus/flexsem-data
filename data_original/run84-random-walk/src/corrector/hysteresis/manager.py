# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import json
import logging
from collections import deque

import numpy as np
from flexsem.utils import Distance

from corrector.config.models import HysteresisConfig
from corrector.core.coordinates import StageOffset, StagePosition
from corrector.hysteresis.const import *

logger = logging.getLogger(__name__)


class HysteresisManager:
    def __init__(self, hysteresis_cfg: HysteresisConfig):
        self.enabled = hysteresis_cfg.enabled
        self.force_warmup = hysteresis_cfg.force_warmup
        self.model_dir = hysteresis_cfg.model_path
        self.model_path = self.model_dir / MODEL_PATH
        self.scaler_path = self._resolve_existing_path(
            self.model_dir / SCALER_PATH,
            self.model_dir / SCALER_FALLBACK_PATH,
        )
        self.config_path = self.model_dir / CONFIG_PATH
        self.simulator = hysteresis_cfg.simulator
        self.sanity_limit = hysteresis_cfg.sanity_limit.value
        self.history = deque()

        if not self.enabled:
            self.window_size = 0
            logger.info("Hysteresis disabled in cfg.yaml")

            if self.force_warmup:
                logger.info(
                    "Hysteresis disabled, but warmup sequence will still be executed"
                )
            else:
                # Compensation nor warmup -> stop init
                return

        self.model_config = self._load_model_config()
        self.inverse_model = bool(self.model_config["inverse_model"])
        self.window_coord_mode = self.model_config["window_coord_mode"]
        self.target_mode = self.model_config["target_mode"]

        import joblib
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        logger.info(f"Loaded ONNX model from {self.model_path}")

        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[2]
        self.window_size = input_shape[1]
        self._validate_model_config()

        logger.info(
            f"Hysteresis window size: {self.window_size}, input size: {self.input_size}, "
            f"window_coord_mode: {self.window_coord_mode}, target_mode: {self.target_mode}"
        )

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

    def should_warmup(self) -> bool:
        return self.enabled or self.force_warmup

    @staticmethod
    def _resolve_existing_path(*paths):
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    def _load_model_config(self):
        config = {
            "inverse_model": INVERSE_MODEL,
            "window_coord_mode": WINDOW_COORD_MODE,
            "target_mode": TARGET_MODE,
        }

        if not self.config_path.exists():
            logger.warning(
                f"Hysteresis model config not found at {self.config_path}; "
                "using fallback values from corrector.hysteresis.const"
            )
            return config

        with open(self.config_path, "r", encoding="utf-8") as handle:
            file_config = json.load(handle)

        config.update(file_config)
        logger.info(f"Loaded hysteresis model config from {self.config_path}")
        return config

    def _validate_model_config(self):
        if self.input_size not in (2, 4):
            raise ValueError(
                f"Unsupported hysteresis model input size: {self.input_size}"
            )

        if self.window_coord_mode not in ("relative", "delta"):
            raise ValueError(f"Unsupported WINDOW_COORD_MODE: {self.window_coord_mode}")

        if self.target_mode not in ("actual_delta", "residual_delta"):
            raise ValueError(f"Unsupported TARGET_MODE: {self.target_mode}")

        if self.target_mode == "residual_delta" and self.window_coord_mode != "delta":
            raise ValueError(
                "TARGET_MODE='residual_delta' is only supported for delta models"
            )

        configured_input_size = self.model_config.get("input_size")
        if (
            configured_input_size is not None
            and int(configured_input_size) != self.input_size
        ):
            raise ValueError(
                f"Model config input_size={configured_input_size} does not match ONNX input size {self.input_size}"
            )

        configured_sequence_length = self.model_config.get("sequence_length")
        if (
            configured_sequence_length is not None
            and isinstance(self.window_size, int)
            and int(configured_sequence_length) != self.window_size
        ):
            raise ValueError(
                f"Model config sequence_length={configured_sequence_length} "
                f"does not match ONNX sequence length {self.window_size}"
            )

    def last_position(self) -> StagePosition | None:
        if len(self.history) == 0:
            return None

        last_entry = self.history[-1]
        return StagePosition(
            x=Distance.from_nanometers(last_entry[2]),
            y=Distance.from_nanometers(last_entry[3]),
        )

    def supports_desired_delta_compensation(self) -> bool:
        """Whether a fresh DIC displacement can be passed directly to the model."""

        return self.enabled and self.inverse_model and self.window_coord_mode == "delta"

    def compensate_desired_delta(
        self,
        desired_delta: StageOffset,
    ) -> StagePosition:
        """Compensate one fresh DIC correction without persistent target bias."""

        if not self.supports_desired_delta_compensation():
            raise ValueError(
                "Desired-delta compensation requires an enabled inverse delta model"
            )

        history_position = self.last_position()
        if history_position is None:
            raise ValueError(
                "Desired-delta compensation requires at least one history observation"
            )

        # The inverse delta model derives its final requested observed movement
        # as model_target - history_position.  Construct a transient target so
        # that this movement equals nominal_position - fresh_DIC_observation.
        model_target = history_position.shifted(desired_delta)
        logger.info(
            "Inverse hysteresis one-shot desired delta dx=%s dy=%s from "
            "history observed x=%s y=%s -> transient model target x=%s y=%s",
            desired_delta.dx,
            desired_delta.dy,
            history_position.x,
            history_position.y,
            model_target.x,
            model_target.y,
        )
        return self.compensate(model_target)

    def _update_history(
        self,
        command_x_nm: float,
        command_y_nm: float,
        observed_x_nm: float,
        observed_y_nm: float,
    ):
        logger.info(
            "Updating hysteresis model history with command=(%s, %s), "
            "DIC observed=(%s, %s)",
            command_x_nm,
            command_y_nm,
            observed_x_nm,
            observed_y_nm,
        )
        self.history.append([command_x_nm, command_y_nm, observed_x_nm, observed_y_nm])

    def _build_absolute_windows(self, candidates: np.ndarray) -> np.ndarray:
        if len(self.history) < self.window_size:
            raise ValueError(
                "History buffer not full. Call initialize_history() first."
            )

        candidates = np.asarray(candidates, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, 2)
        if candidates.shape[1] != 2:
            raise ValueError(
                f"Expected candidates with shape (N, 2), got {candidates.shape}"
            )

        history_arr = np.array(self.history, dtype=float)
        n_candidates = len(candidates)

        if self.input_size == 4:
            base_window = np.empty((self.window_size - 1, self.input_size), dtype=float)
            for i in range(1, self.window_size):
                if self.inverse_model:
                    base_window[i - 1, 0:2] = history_arr[i, 2:4]
                    base_window[i - 1, 2:4] = history_arr[i - 1, 0:2]
                else:
                    base_window[i - 1, 0:2] = history_arr[i, 0:2]
                    base_window[i - 1, 2:4] = history_arr[i - 1, 2:4]

            last_rows = np.empty((n_candidates, 1, self.input_size), dtype=float)
            last_rows[:, 0, 0:2] = candidates
            if self.inverse_model:
                last_rows[:, 0, 2:4] = history_arr[-1, 0:2]
            else:
                last_rows[:, 0, 2:4] = history_arr[-1, 2:4]
        else:
            if self.inverse_model:
                base_window = history_arr[1:, 2:4]
            else:
                base_window = history_arr[1:, 0:2]
            last_rows = candidates[:, np.newaxis, :]

        base_tiled = np.broadcast_to(
            base_window[np.newaxis, :, :],
            (n_candidates, self.window_size - 1, self.input_size),
        ).copy()
        return np.concatenate([base_tiled, last_rows], axis=1)

    def _to_model_coordinate_windows(self, windows_abs: np.ndarray) -> np.ndarray:
        if self.window_coord_mode == "relative":
            ref = windows_abs[:, 0, 0:2]
            if self.input_size == 4:
                ref_full = np.concatenate([ref, ref], axis=1)[:, np.newaxis, :]
            else:
                ref_full = ref[:, np.newaxis, :]
            return windows_abs - ref_full

        if self.window_coord_mode == "delta":
            windows_delta = np.zeros_like(windows_abs)
            windows_delta[:, 1:, :] = np.diff(windows_abs, axis=1)
            return windows_delta

        raise ValueError(f"Unsupported WINDOW_COORD_MODE: {self.window_coord_mode}")

    def _scale_windows(self, windows_nm: np.ndarray) -> np.ndarray:
        shape_orig = windows_nm.shape
        return (
            self.scaler.transform(windows_nm.reshape(-1, 1))
            .reshape(shape_orig)
            .astype(np.float32)
        )

    def _inverse_scale_predictions(self, preds_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1, 2)

    def _last_observed_positions(self, n_rows: int) -> np.ndarray:
        last_observed = np.array(self.history[-1][2:4], dtype=float)
        return np.broadcast_to(last_observed, (n_rows, 2))

    def _last_command_positions(self, n_rows: int) -> np.ndarray:
        last_command = np.array(self.history[-1][0:2], dtype=float)
        return np.broadcast_to(last_command, (n_rows, 2))

    def _predictions_to_observed_positions(
        self,
        preds_nm: np.ndarray,
        windows_abs: np.ndarray,
    ) -> np.ndarray:
        if self.window_coord_mode == "relative":
            return preds_nm + windows_abs[:, 0, 0:2]

        if self.window_coord_mode == "delta":
            if self.target_mode == "actual_delta":
                observed_delta = preds_nm
            elif self.target_mode == "residual_delta":
                command_delta = windows_abs[:, -1, 0:2] - windows_abs[:, -2, 0:2]
                observed_delta = preds_nm + command_delta
            else:
                raise ValueError(f"Unsupported TARGET_MODE: {self.target_mode}")

            return self._last_observed_positions(len(windows_abs)) + observed_delta

        raise ValueError(f"Unsupported WINDOW_COORD_MODE: {self.window_coord_mode}")

    def _predictions_to_command_positions(
        self,
        preds_nm: np.ndarray,
        windows_abs: np.ndarray,
    ) -> np.ndarray:
        if self.window_coord_mode == "relative":
            return preds_nm + windows_abs[:, 0, 0:2]

        if self.window_coord_mode == "delta":
            if self.target_mode == "actual_delta":
                command_delta = preds_nm
            elif self.target_mode == "residual_delta":
                observed_delta = windows_abs[:, -1, 0:2] - windows_abs[:, -2, 0:2]
                command_delta = preds_nm + observed_delta
            else:
                raise ValueError(f"Unsupported TARGET_MODE: {self.target_mode}")

            return self._last_command_positions(len(windows_abs)) + command_delta

        raise ValueError(f"Unsupported WINDOW_COORD_MODE: {self.window_coord_mode}")

    def _run_model(self, windows_abs: np.ndarray) -> np.ndarray:
        windows_model_nm = self._to_model_coordinate_windows(windows_abs)
        windows_scaled = self._scale_windows(windows_model_nm)
        preds_scaled = self.session.run(
            [self.output_name], {self.input_name: windows_scaled}
        )[0]
        preds_nm = self._inverse_scale_predictions(preds_scaled)
        if self.inverse_model:
            return self._predictions_to_command_positions(preds_nm, windows_abs)
        return self._predictions_to_observed_positions(preds_nm, windows_abs)

    def _prepare_input(self, test_cmd_x, test_cmd_y):
        windows_abs = self._build_absolute_windows(
            np.array([[test_cmd_x, test_cmd_y]], dtype=float)
        )
        windows_model_nm = self._to_model_coordinate_windows(windows_abs)
        return self._scale_windows(windows_model_nm)

    def _predict_observed_position(self, cmd_x, cmd_y):
        windows_abs = self._build_absolute_windows(
            np.array([[cmd_x, cmd_y]], dtype=float)
        )
        pred_abs = self._run_model(windows_abs)
        return pred_abs[0, 0], pred_abs[0, 1]

    def compensate(self, target: StagePosition) -> StagePosition:
        if not self.enabled:
            return target

        command, model_error = self.find_optimal_command(target)

        if not self.inverse_model and model_error > self.sanity_limit:
            logger.warning(
                f"Hysteresis compensation distance {model_error}nm "
                f"exceeds sanity limit {self.sanity_limit}nm, skipping compensation"
            )
            return target

        if self.inverse_model:
            logger.info(
                f"Inverse hysteresis compensation applied: target "
                f"x={target.x} y={target.y} -> command "
                f"x={command.x} y={command.y}"
            )
        else:
            logger.info(
                f"Hysteresis compensation applied: target "
                f"x={target.x} y={target.y} -> command "
                f"x={command.x} y={command.y} "
                f"(dist={model_error}nm)"
            )
        return command

    def record_movement(
        self,
        command: StagePosition,
        observed_position: StagePosition,
        confidence: float,
    ) -> None:
        logger.info(
            "Recording hysteresis movement command=(%s, %s), "
            "DIC observed=(%s, %s), confidence=%.3f",
            command.x,
            command.y,
            observed_position.x,
            observed_position.y,
            confidence,
        )

        self._update_history(
            command.x.nanometers,
            command.y.nanometers,
            observed_position.x.nanometers,
            observed_position.y.nanometers,
        )

    # @TODO: revise or remove
    def find_optimal_command(
        self,
        target: StagePosition,
    ) -> tuple[StagePosition, Distance]:
        if not self.enabled:
            return target, Distance.from_nanometers(0)

        target_x_nm = target.x.nanometers
        target_y_nm = target.y.nanometers

        if len(self.history) < self.window_size:
            logger.error(
                "Hysteresis history not full. Call initialize_history() first."
            )

        # Inverse model path (direct prediction)
        if self.inverse_model:
            target = np.array([[target_x_nm, target_y_nm]], dtype=float)
            full_windows = self._build_absolute_windows(target)
            pred_cmd_tensor = self._run_model(full_windows)[0]
            pred_cmd = StagePosition(
                x=Distance.from_nanometers(int(round(pred_cmd_tensor[0]))),
                y=Distance.from_nanometers(int(round(pred_cmd_tensor[1]))),
            )

            logger.info(
                f"Inverse hysteresis compensation from "
                f"(x={target_x_nm} nm, y={target_y_nm} nm) to "
                f"(x={pred_cmd.x.nanometers} nm, y={pred_cmd.y.nanometers} nm)"
            )
            return pred_cmd, Distance.from_nanometers(0)

        # Forward model path (multiple candidate selection)
        candidates = []
        for x_mul in SEARCH_RANGE:
            for y_mul in SEARCH_RANGE:
                candidates.append(
                    [
                        target_x_nm + x_mul * STEP_NM,
                        target_y_nm + y_mul * STEP_NM,
                    ]
                )
        candidates = np.array(candidates, dtype=float)
        n_candidates = len(candidates)

        full_windows = self._build_absolute_windows(candidates)
        preds_abs = self._run_model(full_windows)

        dists = np.sqrt(np.sum((preds_abs - [target_x_nm, target_y_nm]) ** 2, axis=1))
        best_idx = np.argmin(dists)
        min_error = dists[best_idx]

        best_cmd = StagePosition(
            x=Distance.from_nanometers(int(candidates[best_idx, 0])),
            y=Distance.from_nanometers(int(candidates[best_idx, 1])),
        )

        logger.info(
            f"Hysteresis compensation from "
            f"(x={target_x_nm} nm, y={target_y_nm} nm) to "
            f"(x={best_cmd.x.nanometers} nm, y={best_cmd.y.nanometers} nm) "
            f"with dist={min_error:.1f}nm ({n_candidates} candidates)"
        )
        return best_cmd, Distance.from_nanometers(int(min_error))
