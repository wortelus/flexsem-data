# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
import time
from copy import deepcopy

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMScanMode
from flexsem.config.models import Settings as FlexSEMSettings
from flexsem.state.manager import CommunicationManager
from flexsem.utils.metric import Distance, Rotation

from corrector.config.metric import XYR
from corrector.config.models import Settings
from corrector.config.sanity import sanity
from corrector.core.scan import correction_scan, scan
from corrector.core.task import TaskType, build_task_list

logger = logging.getLogger(__name__)


def get_current_position(
    cfg: Settings, manager: CommunicationManager, factory: MessageFactory
) -> dict | None:
    """
    Fetches and returns the current stage position (X, Y, Z, T, R) in nanometers/microradians.
    """
    try:
        logger.debug("Fetching current stage position (get_stage_xyztr)...")
        msg = factory.get_stage_xyztr()
        response = manager.send_command_blocking(
            msg, timeout=cfg.misc.command_timeout_ms
        )

        if response:
            pos_data = response.split(",")
            if len(pos_data) != 5:
                logger.error(
                    f"Unexpected position data format: {response}, expected 5 values"
                )
                return None
            position = {
                "x": Distance.from_nanometers(int(pos_data[0])),
                "y": Distance.from_nanometers(int(pos_data[1])),
                "z": Distance.from_nanometers(int(pos_data[2])),
                "t": Distance.from_nanometers(int(pos_data[3])),
                "r": Rotation.from_degrees(int(pos_data[4])),
            }
            return position
        else:
            logger.error("Failed to fetch position, the response contains no data")
            return None
    except Exception as e:
        logger.error(f"Error while fetching position: {e}")
        return None


def run(cfg: Settings, sem_cfg: FlexSEMSettings):
    logger.info("Starting flexsem-corrector")
    for i, step in enumerate(cfg.exposure.steps):
        logger.info(
            f"Step {i + 1}: Exposure time {step.exposure_time_s}s at mag {step.exposure_mag or cfg.drift.exposure_mag}, "
            f"Correction mag {step.correction_mag or cfg.drift.correction_mag}"
        )

    # Run basic sanity checks
    sanity(cfg)

    # Command factory and connection manager for SEM
    cmd_factory = MessageFactory(
        server_uid=sem_cfg.server.unit_id, client_uid=sem_cfg.client.unit_id
    )

    manager = CommunicationManager(sem_cfg)
    manager.connect()
    logger.info(f"Connected to {sem_cfg.server.unit_id} SEM device")

    tasks = build_task_list(cfg, cmd_factory, manager)
    logger.info(f"Built task list with {len(tasks)} tasks")

    ref_img_path = None
    drift: XYR = XYR.zeroes()
    for i, task in enumerate(tasks):
        logger.info(f"Executing task {i + 1}/{len(tasks)}: {task.task_type}")

        # Move to predefined position + drift
        # step = task.step
        # if i <= 40:
        #     y_task = step.xyr.y.value - Distance.from_nanometers(100 * i)
        # elif 40 < i <= 80:
        #     y_task = step.xyr.y.value - Distance.from_nanometers(100 * 40) + Distance.from_nanometers(100 * (i - 40))
        # elif 80 < i <= 120:
        #     y_task = step.xyr.y.value + Distance.from_nanometers(100 * (i - 80))
        # else:
        #     y_task = step.xyr.y.value + Distance.from_nanometers(100 * 40) - Distance.from_nanometers(100 * (i - 120))
        # x_task = step.xyr.x.value
        # r_task = step.xyr.r.val
        #
        # bl_cancel = cmd_factory.backlash_cancel(Distance.from_nanometers(4000))
        # manager.send_command_blocking(bl_cancel, timeout=cfg.misc.command_timeout_ms)
        # logger.info("Backlash compensation command sent")

        # Move to position
        # set_stage_xyr = cmd_factory.set_stage_xyr(x=x_task, y=y_task, r=r_task)
        # manager.send_command_blocking(
        #     set_stage_xyr, timeout=cfg.misc.command_timeout_ms
        # )
        # logger.info(f"Moved the stage x={x_task} y={y_task} r={r_task}")

        # bl_cancel = cmd_factory.backlash_cancel(Distance.from_nanometers(4000))
        # manager.send_command_blocking(bl_cancel, timeout=cfg.misc.command_timeout_ms)
        # logger.info("Backlash compensation command sent")

        # Get actual position (ex. FlexSEM 1000 has 50nm resolution)
        actual_pos_response = get_current_position(cfg, manager, cmd_factory)
        if actual_pos_response is None:
            logger.error("Failed to fetch current position, aborting")
            return

        # Update current position with actual position
        x_actual = actual_pos_response["x"]
        y_actual = actual_pos_response["y"]
        r_actual = actual_pos_response["r"]
        logger.info(f"Actual position x={x_actual} y={y_actual} r={r_actual}")

        time.sleep(5)  # Wait for the stage to settle
        scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=f"after_move_{i}")

        (dx, dy, conf), ref_img_path_new = task.execute(ref_img_path=ref_img_path)
        dx = -dx

        if task.task_type == TaskType.UPDATE_REFERENCE:
            # Update reference image if needed
            logger.info(
                f"Updated reference image from {ref_img_path} to {ref_img_path_new}"
            )
            ref_img_path = ref_img_path_new
            ref_meta_path = ref_img_path_new.with_suffix(".txt")
        elif task.task_type == TaskType.CORRECTION:
            if conf >= cfg.drift.min_confidence:
                # Update drift value
                x_sem_round_offset = x_actual - step.xyr.x.value
                y_sem_round_offset = y_actual - step.xyr.y.value

                drift.x.value = -dx
                drift.y.value = -dy
                logger.info(
                    f"Drift relative to actual stage position: dx={dx}, dy={dy} with confidence {conf:.3f}"
                )
            else:
                logger.warning(
                    f"Low confidence {conf:.3f} in drift calculation, skipping drift update"
                )

        logger.info(f"Accumulated offset: {drift}")
