# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import numpy as np
from scipy import signal


def create_radial_mask(image, inner_radius, outer_radius):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask = np.ones((height, width), dtype=np.float32)
    mask[dist_from_center < inner_radius] = 0.0

    transition_zone = (dist_from_center >= inner_radius) & (
        dist_from_center <= outer_radius
    )
    mask[transition_zone] = (dist_from_center[transition_zone] - inner_radius) / (
        outer_radius - inner_radius
    )

    return mask


def tukey_window(img, tukey_alpha=0.25):
    # Default rolloff is set to default 0.25
    # Ref: Snella, Michael T. Drift Correction for Scanning-Electron Microscopy. 2010. Massachusetts Institute of Technology, Master of Engineering thesis.
    rows, cols = img.shape

    # Tukey window
    tukey_r = signal.windows.tukey(rows, alpha=tukey_alpha)
    tukey_c = signal.windows.tukey(cols, alpha=tukey_alpha)
    tukey_window_2d = np.outer(tukey_r, tukey_c)
    img_windowed = img * tukey_window_2d

    return img_windowed
