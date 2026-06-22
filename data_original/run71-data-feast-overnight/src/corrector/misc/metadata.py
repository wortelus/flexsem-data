# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import configparser
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from flexsem.utils import Distance


def load_metadata(path: Path):
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-16")
    return config["SemImageFile"]


def get_metric_pixel(
    pixels: tuple[float, float], metadata_config: configparser.SectionProxy
) -> tuple[Distance, Distance]:
    pixel_size = float(metadata_config["PixelSize"])
    dx = pixels[0] * pixel_size
    dy = pixels[1] * pixel_size

    # @TODO: avoid this - it would be better to store it as a float, even if SEM accepts only int nanometers
    x = Distance.from_nanometers(int(Decimal(dx).quantize(0, rounding=ROUND_HALF_UP)))
    y = Distance.from_nanometers(int(Decimal(dy).quantize(0, rounding=ROUND_HALF_UP)))

    return x, y


def get_width_height(metadata_config: configparser.SectionProxy) -> tuple[int, int]:
    datasize = str(metadata_config["DataSize"])
    w, h = datasize.split("x")
    return int(w), int(h)
