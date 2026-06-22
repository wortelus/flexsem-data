# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from corrector.config.models import Settings


def sanity(cfg: Settings):
    sanity_mkdir(cfg)
    sanity_focus(cfg)


def sanity_mkdir(cfg: Settings):
    cfg.misc.temp_path.mkdir(parents=True, exist_ok=True)


def sanity_focus(cfg: Settings):
    magnification_values = set([mag for mag in cfg.focus.keys()])

    # Correction magnification
    if cfg.drift.correction_mag not in magnification_values:
        raise ValueError(
            f"Focus settings for correction mag {cfg.drift.correction_mag} not found in focus configuration."
        )

    # Default exposure magnification
    if cfg.exposure.exposure_mag not in magnification_values:
        raise ValueError(
            f"Focus settings for drift exposure mag {cfg.exposure.exposure_mag} not found in focus configuration."
        )

    # Override exposure magnifications
    for step in cfg.exposure.steps:
        mag_step = step.exposure_mag
        if mag_step not in magnification_values:
            raise ValueError(
                f"Focus settings for exposure mag {mag_step} not found in focus configuration."
            )
