# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from pathlib import Path

from flexsem.utils import Distance as SEMDistance

from corrector.alg.match_template import (
    create_template_matching_mask,
    match_template_with_mask,
    preprocess_template_matching,
)
from corrector.config.models import Settings
from corrector.misc.metadata import get_metric_pixel, load_metadata
from corrector.preprocessing.load import load_grayscale

logger = logging.getLogger(__name__)


def dic_measure(
    cfg: Settings,
    reference_image_path: Path,
    current_image_path: Path,
    current_metadata_path: Path,
    override_mask_disable: bool = False,
) -> tuple[tuple[SEMDistance, SEMDistance], float]:
    """Measure current image-content displacement from the reference image.

    The returned pair uses OpenCV image coordinates: positive X points right
    and positive Y points down.  This function never converts to SEM stage
    coordinates.
    """

    # Load images (current, previous) & metadata (current)
    img_ref, img_current = (
        load_grayscale(reference_image_path.as_posix(), normalize=False),
        load_grayscale(current_image_path.as_posix(), normalize=False),
    )

    metadata_current = load_metadata(current_metadata_path)

    # Preprocess images
    img_preprocess_ref, img_preprocess_current = (
        preprocess_template_matching(img_ref),
        preprocess_template_matching(img_current),
    )

    if not override_mask_disable:
        logger.info("Complete template matching mask applied")
        mask = create_template_matching_mask(
            img_preprocess_ref.shape,
            ignore_center_x=cfg.drift.ignore_center_x_pix,
            ignore_center_y=cfg.drift.ignore_center_y_pix,
            bottom_border=cfg.drift.ignore_bottom_pix,
        )
    else:
        logger.info("Mask is ignoring borders only")
        mask = create_template_matching_mask(
            img_preprocess_ref.shape,
            ignore_center_x=0,
            ignore_center_y=0,
            bottom_border=cfg.drift.ignore_bottom_pix,
        )

    # The algorithm normalizes its alignment transform to the public DIC
    # displacement convention before returning it.
    image_shift_pixels, conf = match_template_with_mask(
        img_preprocess_ref,
        img_preprocess_current,
        mask,
    )
    logger.info(f"DIC image displacement={image_shift_pixels} conf={conf}")

    return get_metric_pixel(image_shift_pixels, metadata_current), conf
