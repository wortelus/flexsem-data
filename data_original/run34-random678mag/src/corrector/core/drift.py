# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import configparser
import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from flexsem.commands import MessageFactory
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Distance, Rotation

from corrector.alg.phase import (
    load_phase_correlation,
    phase_correlation,
    preprocess_phase_correlation,
)
from corrector.core.mask import apply_radial_mask
from corrector.misc.metadata import load_metadata, get_metric_pixel
from src.corrector.config.models import Settings

logger = logging.getLogger(__name__)


def calculate_drift(
    cfg: Settings,
    ref_img_path_current: Path,
    ref_img_path_prev: Path,
    exp_metadata: configparser.SectionProxy,
):
    # Load images (current, previous) & metadata (current)
    img_current, img_prev = (
        load_phase_correlation(ref_img_path_current),
        load_phase_correlation(ref_img_path_prev),
    )
    metadata_cur = load_metadata(ref_img_path_current.with_suffix(".txt"))
    metadata_prev = load_metadata(ref_img_path_prev.with_suffix(".txt"))

    # Preprocess images
    ref_img_proc_current, ref_img_proc_prev = (
        preprocess_phase_correlation(img_current),
        preprocess_phase_correlation(img_prev),
    )

    # Apply radial mask
    if cfg.drift.ignore_center:
        logger.info(f"Applying center mask")
        ref_img_proc_current = apply_radial_mask(cfg, ref_img_proc_current, exp_metadata, metadata_cur)
        ref_img_proc_prev = apply_radial_mask(cfg, ref_img_proc_prev, exp_metadata, metadata_prev)

    # Calculate shift
    shift, conf = phase_correlation(ref_img_proc_current, ref_img_proc_prev)
    logger.info(f"Phase correlation: shift={shift} conf={conf}")

    # Convert to micrometers
    return get_metric_pixel(shift, metadata_cur), conf


def stage_drift(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    new_position: tuple[Distance, Distance],
):
    xn, yn = new_position

    # Apply drift correction
    x = Distance.from_nanometers(int(Decimal(xn.nanometers).quantize(0, rounding=ROUND_HALF_UP)))
    y = Distance.from_nanometers(int(Decimal(yn.nanometers).quantize(0, rounding=ROUND_HALF_UP)))
    r = Rotation.from_degrees(0)

    set_stage_xy = cmd_factory.set_stage_xyr(x, y, r)
    manager.send_command_blocking(set_stage_xy, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Applied drift correction (x, y, r): ({x}, {y})")
