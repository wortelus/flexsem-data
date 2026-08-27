# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

from __future__ import annotations

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
import math
import time

from flexsem.commands import MessageFactory
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Rotation as SEMRotation

from corrector.config.models import ExposureStepConfig, Settings
from corrector.core.coordinates import StagePosition
from corrector.core.dic import dic_measure
from corrector.core.measurement import (
    PositionObservation,
    ReferenceFrame,
    dic_to_stage_offset,
    position_from_dic,
)
from corrector.core.positioning import (
    PositionController,
    PositioningError,
    PrecisionThresholdError,
)
from corrector.core.scan import correction_scan, exposure_scan
from corrector.utils.sem import (
    log_reported_motor_position,
    send_command_reliable_cfg,
)

logger = logging.getLogger(__name__)


class BaseTask:
    """Shared hardware and DIC primitives for positioning tasks."""

    settle_time_s = 3.0
    override_mask_disable = False

    def __init__(
        self,
        cfg: Settings,
        cmd_factory: MessageFactory,
        manager: CommunicationManager,
        step: ExposureStepConfig,
    ):
        self.cfg = cfg
        self.cmd_factory = cmd_factory
        self.manager = manager
        self.step = step

    def _move_to(
        self,
        motor_command: StagePosition,
        rotation: SEMRotation,
    ) -> None:
        set_stage_xyr = self.cmd_factory.set_stage_xyr(
            x=motor_command.x,
            y=motor_command.y,
            r=rotation,
        )
        send_command_reliable_cfg(
            set_stage_xyr,
            cfg=self.cfg,
            cmd_factory=self.cmd_factory,
            manager=self.manager,
        )
        logger.info(
            "Moved stage using motor command x=%s y=%s r=%s",
            motor_command.x,
            motor_command.y,
            rotation,
        )

    def _measure(self, reference: ReferenceFrame) -> PositionObservation:
        try:
            image_path, metadata_path = correction_scan(
                self.cfg,
                self.cmd_factory,
                self.manager,
                settle_time_s=self.settle_time_s,
            )
            (image_dx, image_dy), confidence = dic_measure(
                self.cfg,
                reference.image_path,
                image_path,
                metadata_path,
                override_mask_disable=self.override_mask_disable,
            )

            # TODO: refine the low-confidence/residual policy.  Rejecting this
            # observation does not roll back any earlier valid movements.
            if (
                not math.isfinite(confidence)
                or confidence < self.cfg.drift.min_confidence
            ):
                raise PositioningError(
                    f"DIC confidence {confidence!r} is below required "
                    f"{self.cfg.drift.min_confidence}"
                )

            stage_offset = dic_to_stage_offset(image_dx, image_dy)
            observed_position = position_from_dic(reference, stage_offset)
        except PositioningError:
            raise
        except Exception as exc:
            raise PositioningError(
                f"Correction scan or DIC measurement failed: {exc}"
            ) from exc
        logger.info(
            "DIC observation relative to site %s: image_dx=%s image_dy=%s, "
            "stage_dx=%s stage_dy=%s, observed_x=%s observed_y=%s, "
            "confidence=%.3f",
            reference.site_id,
            image_dx,
            image_dy,
            stage_offset.dx,
            stage_offset.dy,
            observed_position.x,
            observed_position.y,
            confidence,
        )
        return PositionObservation(
            observed_position=observed_position,
            confidence=confidence,
            image_path=image_path,
            metadata_path=metadata_path,
        )

    def move_and_measure(
        self,
        reference: ReferenceFrame,
        motor_command: StagePosition,
        rotation: SEMRotation,
    ) -> PositionObservation:
        try:
            self._move_to(motor_command, rotation)
        except PositioningError:
            raise
        except Exception as exc:
            raise PositioningError(f"Stage movement failed: {exc}") from exc

        log_reported_motor_position(
            self.cfg,
            self.manager,
            self.cmd_factory,
            context="Post-movement",
        )
        return self._measure(reference)

    def measure_current(
        self,
        reference: ReferenceFrame,
    ) -> PositionObservation:
        return self._measure(reference)


class UpdateReferenceTask(BaseTask):
    settle_time_s = 5.0
    override_mask_disable = True

    def run(
        self,
        source_reference: ReferenceFrame,
        nominal_position: StagePosition,
        rotation: SEMRotation,
        controller: PositionController,
    ) -> PositionObservation:
        final_observation = None
        motor_command = controller.command_for(nominal_position)
        for movement_number in (1, 2):
            final_observation = self.move_and_measure(
                source_reference,
                motor_command,
                rotation,
            )
            controller.record_movement(
                motor_command,
                final_observation,
            )
            position_error = controller.position_error(
                nominal_position,
                final_observation.observed_position,
            )
            logger.info(
                "Reference update movement %d/2 residual error: dx=%s dy=%s",
                movement_number,
                position_error.dx,
                position_error.dy,
            )
            # Only the first residual feeds the second command.  A future
            # correction starts from a fresh DIC observation instead of
            # persisting this task's final residual.
            if movement_number == 1:
                motor_command = controller.corrective_command_for(
                    nominal_position,
                    final_observation.observed_position,
                )

        return final_observation


class PreciseUpdateReferenceTask(UpdateReferenceTask):
    def run(
        self,
        source_reference: ReferenceFrame,
        nominal_position: StagePosition,
        rotation: SEMRotation,
        controller: PositionController,
    ) -> PositionObservation:
        precise_cfg = self.cfg.update_reference.precise
        threshold_nm = precise_cfg.distance_threshold.value.nanometers

        motor_command = controller.command_for(nominal_position)
        for attempt in range(1, precise_cfg.max_attempts + 1):
            observation = self.move_and_measure(
                source_reference,
                motor_command,
                rotation,
            )
            controller.record_movement(
                motor_command,
                observation,
            )
            position_error = controller.position_error(
                nominal_position,
                observation.observed_position,
            )
            distance_nm = position_error.distance_nm()
            logger.info(
                "Precise reference update attempt %d/%d: error=%0.1fnm "
                "(dx=%s dy=%s, threshold=%snm)",
                attempt,
                precise_cfg.max_attempts,
                distance_nm,
                position_error.dx,
                position_error.dy,
                threshold_nm,
            )
            if distance_nm <= threshold_nm:
                return observation
            # Build a correction only when another movement will consume it.
            if attempt < precise_cfg.max_attempts:
                motor_command = controller.corrective_command_for(
                    nominal_position,
                    observation.observed_position,
                )

        raise PrecisionThresholdError(
            "Precise reference update did not reach "
            f"{threshold_nm}nm after {precise_cfg.max_attempts} attempts",
            last_observation=observation,
            position_error=position_error,
        )


class CorrectionTask(BaseTask):
    def run(
        self,
        active_reference: ReferenceFrame,
        nominal_position: StagePosition,
        rotation: SEMRotation,
        controller: PositionController,
    ) -> PositionObservation:
        stationary_observation = self.measure_current(active_reference)
        stationary_error = controller.position_error(
            nominal_position,
            stationary_observation.observed_position,
        )
        logger.info(
            "Stationary correction error: dx=%s dy=%s",
            stationary_error.dx,
            stationary_error.dy,
        )

        motor_command = controller.corrective_command_for(
            nominal_position,
            stationary_observation.observed_position,
        )
        final_observation = self.move_and_measure(
            active_reference,
            motor_command,
            rotation,
        )
        controller.record_movement(
            motor_command,
            final_observation,
        )
        final_error = controller.position_error(
            nominal_position,
            final_observation.observed_position,
        )
        # No movement follows.  A later correction remeasures the stationary
        # position instead of persisting this final residual.
        logger.info(
            "Final non-precise correction residual: dx=%s dy=%s",
            final_error.dx,
            final_error.dy,
        )
        return final_observation


class PreciseCorrectionTask(CorrectionTask):
    def run(
        self,
        active_reference: ReferenceFrame,
        nominal_position: StagePosition,
        rotation: SEMRotation,
        controller: PositionController,
    ) -> PositionObservation:
        precise_cfg = self.cfg.drift.precise
        threshold_nm = precise_cfg.distance_threshold.value.nanometers

        observation = self.measure_current(active_reference)
        position_error = controller.position_error(
            nominal_position,
            observation.observed_position,
        )
        if position_error.distance_nm() <= threshold_nm:
            return observation

        motor_command = controller.corrective_command_for(
            nominal_position,
            observation.observed_position,
        )
        for attempt in range(1, precise_cfg.max_attempts + 1):
            observation = self.move_and_measure(
                active_reference,
                motor_command,
                rotation,
            )
            controller.record_movement(
                motor_command,
                observation,
            )
            position_error = controller.position_error(
                nominal_position,
                observation.observed_position,
            )
            distance_nm = position_error.distance_nm()
            logger.info(
                "Precise correction movement %d/%d: error=%0.1fnm "
                "(dx=%s dy=%s, threshold=%snm)",
                attempt,
                precise_cfg.max_attempts,
                distance_nm,
                position_error.dx,
                position_error.dy,
                threshold_nm,
            )
            if distance_nm <= threshold_nm:
                return observation
            if attempt < precise_cfg.max_attempts:
                motor_command = controller.corrective_command_for(
                    nominal_position,
                    observation.observed_position,
                )

        raise PrecisionThresholdError(
            "Precise correction did not reach "
            f"{threshold_nm}nm after {precise_cfg.max_attempts} corrective "
            "movements",
            last_observation=observation,
            position_error=position_error,
        )


class ExposureTask:
    """An exposure scan and timed dwell with no positioning side effects."""

    def __init__(
        self,
        cfg: Settings,
        cmd_factory: MessageFactory,
        manager: CommunicationManager,
        step: ExposureStepConfig,
    ):
        self.cfg = cfg
        self.cmd_factory = cmd_factory
        self.manager = manager
        self.step = step

    def run(self) -> None:
        logger.info(
            "Executing exposure chunk for site %s (%ss)",
            self.step.site_id,
            self.step.exposure_time_s,
        )
        exposure_scan(
            self.cfg,
            self.cmd_factory,
            self.manager,
            self.step,
        )
        time.sleep(self.step.exposure_time_s)
