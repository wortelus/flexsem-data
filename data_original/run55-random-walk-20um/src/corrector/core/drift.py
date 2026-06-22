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

from corrector.alg.match_template import preprocess_template_matching, match_template_with_mask, \
    create_template_matching_mask
from corrector.alg.phase import (
    load_phase_correlation,
    phase_correlation,
    preprocess_phase_correlation_a,
)
from corrector.core.mask import apply_radial_mask
from corrector.misc.metadata import get_metric_pixel, load_metadata
from corrector.preprocessing.load import load_grayscale
from src.corrector.config.models import Settings

logger = logging.getLogger(__name__)


def calculate_drift(
    cfg: Settings,
    ref_img_path_ref: Path,
    ref_img_path_current: Path,
    exp_metadata: configparser.SectionProxy,
    override_mask_disable: bool = False,
):
    # Load images (current, previous) & metadata (current)
    img_ref, img_current = (
        load_grayscale(ref_img_path_ref.as_posix(), normalize=False),
        load_grayscale(ref_img_path_current.as_posix(), normalize=False),
    )

    metadata_current = load_metadata(ref_img_path_current.with_suffix(".txt"))

    # Preprocess images
    img_preprocess_ref, img_preprocess_current = (
        preprocess_template_matching(img_ref),
        preprocess_template_matching(img_current),
    )

    # Apply radial mask
    # @TODO: fix
    # if cfg.drift.ignore_center:
    #     logger.info(f"Applying center mask")
    #     ref_img_proc_current = apply_radial_mask(
    #         cfg, ref_img_proc_current, exp_metadata, metadata_cur
    #     )
    #     ref_img_proc_prev = apply_radial_mask(
    #         cfg, ref_img_proc_prev, exp_metadata, metadata_prev
    #     )
    if not override_mask_disable:
        logger.info(f"Complete template matching mask applied")
        mask = create_template_matching_mask(img_preprocess_ref.shape)
    else:
        logger.info(f"Mask is ignoring borders only")
        mask = create_template_matching_mask(img_preprocess_ref.shape, ratio=0.0)

    # Calculate shift
    shift, conf = match_template_with_mask(img_preprocess_ref, img_preprocess_current, mask)
    logger.info(f"DIC shift={shift} conf={conf} (TL origin)")

    # Convert to micrometers
    return get_metric_pixel(shift, metadata_current), conf


def stage_drift(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    new_position: tuple[Distance, Distance],
):
    xn, yn = new_position

    # Apply drift correction
    x = Distance.from_nanometers(
        int(Decimal(xn.nanometers).quantize(0, rounding=ROUND_HALF_UP))
    )
    y = Distance.from_nanometers(
        int(Decimal(yn.nanometers).quantize(0, rounding=ROUND_HALF_UP))
    )
    r = Rotation.from_degrees(0)

    set_stage_xy = cmd_factory.set_stage_xyr(x, y, r)
    manager.send_command_blocking(set_stage_xy, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Applied drift correction (x, y, r): ({x}, {y})")
