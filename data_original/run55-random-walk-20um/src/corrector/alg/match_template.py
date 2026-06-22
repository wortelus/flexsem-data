# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def match_template(img, template, mask=None):
    """
    Perform template matching using OpenCV's matchTemplate function.
    :param img: Input image (grayscale, float32).
    :param template: Template image (grayscale, float32).
    :param mask: Mask to ignore certain regions (same size as img).
    :return: (xy, max_val)
    """
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def preprocess_template_matching(img, ksize: int = 27):
    # Median filter
    img_denoised = cv2.medianBlur(img, ksize=ksize)
    return img_denoised


def match_template_with_mask(img, template, mask=None, padding: int = 512, border: int = 40):
    """
    Perform template matching using OpenCV's matchTemplate function with a mask.
    :param img: Input image (grayscale, float32).
    :param template: Template image (grayscale, float32).
    :param mask: Mask to ignore certain regions (same size as img).
    :param padding: Padding size around the image.
    :return: (xy, max_val)
    """
    img_crop = img[border:-border, border:-border]

    mean_val = float(np.mean(img_crop))

    img_padded = cv2.copyMakeBorder(
        img_crop,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=mean_val
    )

    (found_x, found_y), max_val = match_template(img_padded, template, mask=mask)

    dx = found_x - padding + border
    dy = found_y - padding + border

    dx = -dx
    dy = -dy

    return (dx, dy), max_val


def create_template_matching_mask(img_shape,
                                  ratio: float = 0.3,
                                  border: int = 40):
    h, w = img_shape

    mask = np.ones((h, w), dtype=np.float32)

    if border > 0:
        mask[:border, :] = 0.
        mask[-border:, :] = 0.
        mask[:, :border] = 0.
        mask[:, -border:] = 0.

    if ratio > 0:
        cy, cx = h // 2, w // 2
        rh, rw = int(h * ratio / 2), int(w * ratio / 2)
        mask[cy - rh:cy + rh, cx - rw:cx + rw] = 0.

    return mask
