# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import configparser
import logging
from pathlib import Path

from corrector.alg.match_template import preprocess_template_matching, match_template_with_mask, \
    create_template_matching_mask
from corrector.misc.metadata import get_metric_pixel, load_metadata
from corrector.preprocessing.load import load_grayscale
from src.corrector.config.models import Settings

logger = logging.getLogger(__name__)


def dic_measure(
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

    if not override_mask_disable:
        logger.info(f"Complete template matching mask applied")
        mask = create_template_matching_mask(img_preprocess_ref.shape,
                                             ignore_center_x=cfg.drift.ignore_center_x_pix,
                                             ignore_center_y=cfg.drift.ignore_center_y_pix,
                                             bottom_border=cfg.drift.ignore_bottom_pix)
    else:
        logger.info(f"Mask is ignoring borders only")
        mask = create_template_matching_mask(img_preprocess_ref.shape,
                                             ignore_center_x=0,
                                             ignore_center_y=0,
                                             bottom_border=cfg.drift.ignore_bottom_pix)

    # Calculate shift
    shift, conf = match_template_with_mask(img_preprocess_ref, img_preprocess_current, mask)
    logger.info(f"DIC shift={shift} conf={conf}")

    # Convert to micrometers
    return get_metric_pixel(shift, metadata_current), conf
