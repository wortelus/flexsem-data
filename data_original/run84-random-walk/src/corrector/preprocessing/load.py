# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from os.path import exists

import cv2
import numpy as np


def load_grayscale(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load a grayscale image from the specified path in OpenCV.
    """
    if not exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded from {path}")

    if normalize:
        # Normalize the image to the range [0, 1]
        image = np.float32(image) / 255.0

    return image
