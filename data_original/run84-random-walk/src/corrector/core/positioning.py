# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

from __future__ import annotations

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corrector.core.coordinates import StageOffset, StagePosition
from corrector.core.measurement import PositionObservation, ReferenceFrame

if TYPE_CHECKING:
    from corrector.hysteresis.manager import HysteresisManager

logger = logging.getLogger(__name__)


class PositioningError(RuntimeError):
    """A positioning or DIC operation could not produce an accepted result."""


class PrecisionThresholdError(PositioningError):
    """Precise positioning exhausted its attempts with a valid final result."""

    def __init__(
        self,
        message: str,
        last_observation: PositionObservation,
        position_error: StageOffset,
    ):
        super().__init__(message)
        self.last_observation = last_observation
        self.position_error = position_error


@dataclass
class RunState:
    site_references: dict[str, ReferenceFrame]
    active_reference: ReferenceFrame
    feedback_offset: StageOffset

    def reference_for_update(self, site_id: str) -> ReferenceFrame:
        return self.site_references.get(site_id, self.active_reference)

    def commit_reference(
        self,
        site_id: str,
        observation: PositionObservation,
    ) -> ReferenceFrame:
        new_reference = ReferenceFrame(
            site_id=site_id,
            position=observation.observed_position,
            image_path=observation.image_path,
            metadata_path=observation.metadata_path,
        )
        self.site_references.setdefault(site_id, new_reference)
        self.active_reference = new_reference
        return new_reference


class PositionController:
    """Own command bias, position-error math, and movement-history updates."""

    def __init__(self, state: RunState, hysteresis: "HysteresisManager"):
        self._state = state
        self._hysteresis = hysteresis

    def command_for(self, nominal_position: StagePosition) -> StagePosition:
        if self._hysteresis.supports_desired_delta_compensation():
            # The inverse delta model already derives the requested movement
            # from its last DIC observation and this nominal target.
            return self._hysteresis.compensate(nominal_position)

        feedback_adjusted_target = nominal_position.shifted(self._state.feedback_offset)
        return self._hysteresis.compensate(feedback_adjusted_target)

    @staticmethod
    def position_error(
        nominal_position: StagePosition,
        observed_position: StagePosition,
    ) -> StageOffset:
        return StageOffset(
            dx=nominal_position.x - observed_position.x,
            dy=nominal_position.y - observed_position.y,
        )

    def update_feedback(self, position_error: StageOffset) -> None:
        """Apply an error only when a corrective movement will follow."""

        self._state.feedback_offset = self._state.feedback_offset + position_error
        logger.info(
            "Updated positioning feedback offset: dx=%s dy=%s",
            self._state.feedback_offset.dx,
            self._state.feedback_offset.dy,
        )

    def corrective_command_for(
        self,
        nominal_position: StagePosition,
        observed_position: StagePosition,
    ) -> StagePosition:
        """Build a correction from one fresh DIC observation."""

        position_error = self.position_error(nominal_position, observed_position)
        if self._hysteresis.supports_desired_delta_compensation():
            return self._hysteresis.compensate_desired_delta(position_error)

        # Preserve the existing controller for disabled, forward, and relative
        # model modes.  Only inverse delta models consume the fresh error
        # directly and therefore bypass persistent positioning feedback.
        self.update_feedback(position_error)
        return self.command_for(nominal_position)

    def record_movement(
        self,
        motor_command: StagePosition,
        observation: PositionObservation,
    ) -> None:
        self._hysteresis.record_movement(
            command=motor_command,
            observed_position=observation.observed_position,
            confidence=observation.confidence,
        )
