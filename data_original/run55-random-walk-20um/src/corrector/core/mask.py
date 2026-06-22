# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from configparser import SectionProxy

import numpy as np

from corrector.config.models import Settings
from corrector.misc.metadata import get_width_height, get_metric_pixel
from corrector.preprocessing.signal import create_radial_mask


def apply_radial_mask(
    cfg: Settings,
    img_correction: np.ndarray,
    meta_exposure: SectionProxy,
    meta_correction: SectionProxy,
) -> np.ndarray:
    # Total exposure area
    exp_pw, exp_ph = get_width_height(meta_exposure)
    exp_w, exp_h = get_metric_pixel((exp_pw, exp_ph), meta_exposure)

    # Calculate radius in correction image pixels
    corr_pw, corr_ph = get_width_height(meta_correction)
    corr_w, corr_h = get_metric_pixel((corr_pw, corr_ph), meta_correction)

    # Ratio between exposure and correction image
    a = exp_w.micrometers / corr_w.micrometers
    b = exp_h.micrometers / corr_h.micrometers

    # Assume exposure and correction images are centered
    center_px, center_py = corr_pw / 2, corr_ph / 2
    radius_um = np.sqrt(((a * corr_pw) / 2) ** 2 + ((b * corr_ph) / 2) ** 2)

    # Create radial mask
    mask = create_radial_mask(
        img_correction,
        inner_radius=radius_um * cfg.drift.ignore_center_radius_inner_fraction,
        outer_radius=radius_um * cfg.drift.ignore_center_radius_outer_fraction,
    )

    # Apply mask
    img_masked = img_correction * mask

    return img_masked
