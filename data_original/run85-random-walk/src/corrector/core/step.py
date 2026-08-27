# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from corrector.config.models import ExposureStepConfig
from corrector.core.coordinates import StagePosition


def stage_position_from_step(step: ExposureStepConfig) -> StagePosition:
    """Return the configured nominal stage position for an exposure step."""

    return StagePosition(x=step.xyr.x.value, y=step.xyr.y.value)
