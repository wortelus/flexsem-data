# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

from __future__ import annotations

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from dataclasses import dataclass
from pathlib import Path

from flexsem.utils import Distance as SEMDistance

from corrector.core.coordinates import StageOffset, StagePosition


@dataclass(frozen=True)
class ReferenceFrame:
    """An image anchor whose position is expressed in the local DIC frame."""

    site_id: str
    position: StagePosition
    image_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PositionObservation:
    """DIC-derived sample position; never a raw SEM motor readback."""

    observed_position: StagePosition
    confidence: float
    image_path: Path
    metadata_path: Path


def dic_to_stage_offset(
    image_dx: SEMDistance,
    image_dy: SEMDistance,
) -> StageOffset:
    """Convert public OpenCV DIC displacement to SEM stage coordinates once."""

    return StageOffset(dx=-image_dx, dy=image_dy)


def position_from_dic(
    reference: ReferenceFrame,
    stage_offset: StageOffset,
) -> StagePosition:
    """Derive current sample position relative to an immutable image anchor."""

    # TODO: add advanced displacement magnitude/rate validation without
    # promising rollback of movements which have already completed.
    # TODO: define the detailed low-confidence and residual/outlier policy.
    return StagePosition(
        x=reference.position.x - stage_offset.dx,
        y=reference.position.y - stage_offset.dy,
    )
