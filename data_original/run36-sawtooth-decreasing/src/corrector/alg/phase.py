# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from os import makedirs
from os.path import dirname

import cv2
import numpy as np

from corrector.preprocessing.load import load_grayscale
from corrector.preprocessing.signal import tukey_window

logger = logging.getLogger(__name__)


def load_phase_correlation(filename: str):
    return load_grayscale(filename, normalize=False)


def phase_correlation(img_a, img_b):
    """
    Perform phase correlation between two images using OpenCV's built-in function.
    :param img_a: First input image (grayscale, float32).
    :param img_b: Second input image (grayscale, float32).
    :return: (xy, max_val)
    """
    max_loc, max_val = cv2.phaseCorrelate(img_a, img_b)
    xy = (-max_loc[0], -max_loc[1])
    return xy, max_val


def preprocess_phase_correlation(img, ksize: int = 27, tukey_alpha: float = 0.15):
    # Median filter
    img_denoised = cv2.medianBlur(img, ksize=ksize)

    # Sobel filter
    sobelx = cv2.Sobel(img_denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_denoised, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    # 0.5x + 0.5y + 0
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    # Tukey window
    im_final = tukey_window(sobel_combined, tukey_alpha=tukey_alpha)

    return im_final


def phase_correlation_map(img_a, img_b, output_path: str = None):
    """
    Generate a phase correlation map between two images.
    :param img_a: First input image (grayscale, float32).
    :param img_b: Second input image (grayscale, float32).
    :return: Phase correlation map as a 2D numpy array.
    """
    # FFT
    dft_a = cv2.dft(img_a, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_b = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Cross power spectrum
    cross_power_spectrum = cv2.mulSpectrums(
        dft_b, dft_a, flags=cv2.DFT_ROWS, conjB=True
    )

    # Normalization and avoiding division by zero
    mag = cv2.magnitude(cross_power_spectrum[:, :, 0], cross_power_spectrum[:, :, 1])
    mag[mag == 0] = 1e-9

    # 2 channel magnitude (for complex numbers)
    mag_merged = cv2.merge([mag, mag])

    # 2 channel spectrum divided by 2 channel magnitude
    cv2.divide(cross_power_spectrum, mag_merged, cross_power_spectrum)

    # Inverse FFT to get the heatmap
    inverse_dft = cv2.idft(
        cross_power_spectrum, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

    # Shift for intuitive visualization (zero shift in the center)
    heatmap = np.fft.fftshift(inverse_dft)

    # Save the heatmap if output path is provided
    if output_path is not None:
        try:
            # Normalization and coloring for saving
            heatmap_normalized = cv2.normalize(
                heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)

            makedirs(dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, heatmap_colored)
            logger.info(f"Heatmap successfully saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving heatmap image: {e}")

    return heatmap
