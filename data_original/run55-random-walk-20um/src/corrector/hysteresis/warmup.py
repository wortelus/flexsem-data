# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
import random
import time
from math import pi, sin, cos

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMScanMode, SEMSetScanSetting
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Distance, Rotation

from corrector.config.models import Settings
from corrector.core.drift import calculate_drift
from corrector.core.scan import scan
from corrector.hysteresis.manager import HysteresisManager
from corrector.misc.metadata import load_metadata

logger = logging.getLogger(__name__)


def warmup_spiral(seq_length: int,
                  center_x: Distance,
                  center_y: Distance,
                  radius_start: Distance = Distance.from_nanometers(2000),
                  radius_end: Distance = Distance.from_nanometers(3000),
                  radius_random: Distance = Distance.from_nanometers(1000)) -> list[tuple[Distance, Distance]]:
    points = []

    # 1. Spiral
    spiral_len = seq_length
    for i in range(spiral_len):
        # 0 to 4PI (two rounds)
        angle = (i / spiral_len) * 4 * pi

        # Radius in nm
        radius_nm = radius_start.nanometers + (
                (radius_end.nanometers - radius_start.nanometers) *
                (i / (spiral_len - 1))
        )  # Radius grows

        x = center_x.nanometers + radius_nm * cos(angle)
        y = center_y.nanometers + radius_nm * sin(angle)

        points.append((Distance.from_nanometers(int(x)), Distance.from_nanometers(int(y))))

    # # 2. Random movement around center
    # random_len = seq_length // 2
    # for _ in range(random_len):
    #     offset_x = random.uniform(-radius_random.nanometers, radius_random.nanometers)
    #     offset_y = random.uniform(-radius_random.nanometers, radius_random.nanometers)
    #     x = center_x.nanometers + offset_x
    #     y = center_y.nanometers + offset_y
    #     points.append((Distance.from_nanometers(int(x)), Distance.from_nanometers(int(y))))

    # # 2. Linear movement to the end point
    # points_linear = warmup_linear(5, center_x, center_y, radius_end)
    # points.extend(points_linear)

    return points


def warmup_linear(seq_length: int,
                  warmup_end: tuple[Distance, Distance],
                  distance: Distance = Distance.from_nanometers(2000)) -> list[tuple[Distance, Distance]]:
    points = []
    end_x, end_y = warmup_end

    if seq_length <= 1:
        return [(end_x, end_y)]

    for i in range(seq_length):
        # fraction goes from 0 to 1
        # i = 0: farthest from the end point (distance away)
        # i = seq_length - 1: fraction = 1, at the end point
        fraction = i / (seq_length - 1)

        # Start at (end - distance)
        # increase by (distance * fraction)
        x_nm = end_x.nanometers - distance.nanometers * (1 - fraction)
        y_nm = end_y.nanometers + distance.nanometers * (1 - fraction)

        points.append((
            Distance.from_nanometers(int(x_nm)),
            Distance.from_nanometers(int(y_nm))
        ))

    return points


def run_warmup(cfg: Settings,
               manager: CommunicationManager,
               cmd_factory: MessageFactory,
               h_mgr: HysteresisManager,
               points: list):
    logger.info(f"Starting warmup sequence execution")

    # Set initial position
    init_x, init_y = points[0]
    set_stage_xy = cmd_factory.set_stage_xyr(init_x, init_y, Rotation.from_degrees(0))
    manager.send_command_blocking(set_stage_xy, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Stage moved to init position X={init_x.nanometers} nm, Y={init_y.nanometers} nm")

    # Unfreeze
    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    manager.send_command_blocking(set_unfreeze, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Correction scan started, unfrozen")

    # Set magnification
    set_magnification = cmd_factory.set_magnification(cfg.drift.correction_mag)
    manager.send_command_blocking(
        set_magnification, timeout=cfg.misc.command_timeout_ms
    )
    logger.debug(f"Magnification set to {set_magnification.to_wire_format()}")

    # Sleep
    time.sleep(5)

    # Initial scan to get reference image
    scan_type = cfg.drift.correction_type
    ref_img_path, ref_metadata_path = scan(cfg, cmd_factory, manager, scan_type, "warmup_ref")

    # Move through points
    for i, (x, y) in enumerate(points):
        logger.info(f"Warmup step {i + 1}/{len(points)}")

        # Move stage
        set_stage_xy = cmd_factory.set_stage_xyr(x, y, Rotation.from_degrees(0))
        manager.send_command_blocking(set_stage_xy, timeout=cfg.misc.command_timeout_ms)
        logger.debug(f"Stage moved to position X={x.nanometers} nm, Y={y.nanometers} nm")

        # Sleep
        time.sleep(3)

        # Scan
        # TODO: configurable
        warmup_img_filepath, warmup_metadata_filepath = scan(cfg, cmd_factory, manager, scan_type, "warmup")

        metadata = load_metadata(warmup_metadata_filepath)
        (dx, dy), conf = calculate_drift(cfg, ref_img_path, warmup_img_filepath, metadata, override_mask_disable=True)
        dx = -dx  # the stage has TR origin, OpenCV has TL origin
        logger.info(f"Warmup position measured at (x={dx}, y={dy}) with conf={conf} to the original reference image")

        if conf < cfg.drift.min_confidence:
            logger.warning(f"Low confidence ({conf}) in drift calculation, skipping correction")
            continue

        h_mgr.update_history(x.nanometers,
                             y.nanometers,
                             init_x.nanometers - dx.nanometers,
                             init_y.nanometers - dy.nanometers)

    logger.info(f"Warmup sequence execution completed")
