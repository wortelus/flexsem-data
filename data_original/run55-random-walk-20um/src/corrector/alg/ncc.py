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


def find_translation_bruteforce(
    im1_gray, im2_gray, search_x, search_y, ignore_rect=None
):
    im1 = im1_gray.copy().astype(np.float32)
    im2 = im2_gray.copy().astype(np.float32)

    if ignore_rect:
        x, y, w, h = ignore_rect
        cv2.rectangle(im1, (x, y), (x + w, y + h), 0, -1)
        cv2.rectangle(im2, (x, y), (x + w, y + h), 0, -1)

    im1_flat = im1.flatten()

    max_corr = -1.0
    best_shift = (0, 0)

    total_steps = (2 * search_y + 1) * (2 * search_x + 1)
    current_step = 0

    logger.info(f"Launching NCC bruteforce search over {total_steps} positions...")
    for dy in range(-search_y, search_y + 1):
        for dx in range(-search_x, search_x + 1):
            current_step += 1
            if current_step % 1000 == 0:
                logger.info(f"{current_step}/{total_steps}...")

            # Posunutí druhého obrázku
            im2_shifted = np.roll(im2, (dy, dx), axis=(0, 1))
            im2_shifted_flat = im2_shifted.flatten()

            # Výpočet korelačního koeficientu
            # np.corrcoef vrací matici 2x2, nás zajímá hodnota mimo diagonálu
            correlation = np.corrcoef(im1_flat, im2_shifted_flat)[0, 1]

            # Pokud je korelace lepší než dosavadní maximum, uložíme ji
            if correlation > max_corr:
                max_corr = correlation
                best_shift = (dx, dy)

    logger.info(
        f"Bruteforce korelace: nejlepší posun {best_shift}, důvěra (korelace) {max_corr:.4f}"
    )
    return best_shift[0], best_shift[1], max_corr
