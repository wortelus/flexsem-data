# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging

from flexsem.commands import MessageFactory
from flexsem.config.models import Settings as FlexSEMSettings
from flexsem.state.manager import CommunicationManager

from corrector.config.metric import XYR
from corrector.config.models import Settings
from corrector.config.sanity import sanity
from corrector.core.task import TaskType, build_task_list
from corrector.hysteresis.manager import HysteresisManager
from corrector.hysteresis.warmup import run_warmup, warmup_linear
from corrector.utils.sem import get_current_position

logger = logging.getLogger(__name__)


def run(cfg: Settings, sem_cfg: FlexSEMSettings):
    logger.info("Starting flexsem-corrector")
    for i, step in enumerate(cfg.exposure.steps):
        logger.info(
            f"Step {i + 1}: Exposure time {step.exposure_time_s}s at mag {step.exposure_mag or cfg.drift.exposure_mag}, "
            f"Correction mag {cfg.drift.correction_mag}"
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

    warmup_end = cfg.exposure.steps[0].xyr
    warmup_end_xy = warmup_end.x.value, warmup_end.y.value

    h_manager = HysteresisManager(cfg.hysteresis)
    warmup = warmup_linear(h_manager.sequence_length,
                           warmup_end_xy,
                           cfg.hysteresis.warmup_distance.value)
    # warmup = warmup_spiral(h_manager.sequence_length,
    #                        warmup_end.x.value, warmup_end.y.value)
    logger.info(
        f"{len(warmup)} warmup steps generated around x={cfg.hysteresis.warmup_point.x.value} y={cfg.hysteresis.warmup_point.y.value}")

    # @IMPORTANT: warmup must end at the first step position to properly initialize hysteresis history for the first step
    ref_img_path, ref_meta_path = run_warmup(cfg, manager, cmd_factory, h_manager, warmup)
    (ref_x, ref_y) = h_manager.last_position()
    logger.info(f"Warmup completed, reference position set to x={ref_x} y={ref_y}")

    tasks = build_task_list(cfg, cmd_factory, manager)
    logger.info(f"Built task list with {len(tasks)} tasks")

    drift: XYR = XYR.zeroes()
    for i, task in enumerate(tasks):
        logger.info(f"Executing task {i + 1}/{len(tasks)}: {task.task_type}")

        # Predefined position + drift
        step = task.step
        x_task = step.xyr.x.value + drift.x.value
        y_task = step.xyr.y.value + drift.y.value
        r_task = step.xyr.r.val + drift.r.val

        # Predefined position + drift + hysteresis compensation
        (x_task_h, y_task_h), dist = h_manager.find_optimal_command(
            target_x=x_task,
            target_y=y_task,
        )
        if dist > cfg.hysteresis.sanity_limit.value:
            logger.warning(
                f"Hysteresis compensation distance {dist}nm "
                f"exceeds sanity limit {cfg.hysteresis.sanity_limit.value}nm, skipping compensation"
            )
        else:
            logger.info(
                f"Hysteresis compensation applied: target "
                f"x={x_task}nm y={y_task}nm -> command x={x_task_h}nm y={y_task_h}nm "
                f"(dist={dist}nm)"
            )
            x_task, y_task = x_task_h, y_task_h

        # Move to position
        set_stage_xyr = cmd_factory.set_stage_xyr(x=x_task, y=y_task, r=r_task)
        manager.send_command_reliable(
            set_stage_xyr, timeout=cfg.misc.command_timeout_ms
        )
        logger.info(f"Moved the stage x={x_task} y={y_task} r={r_task}")

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

        # We need to scan() before execute(), because execute may block for task.exposure_time_s
        # We want measurement without time drifts
        (dx, dy, conf), current_img_scan_path, current_img_metadata_path = task.scan(ref_img_path=ref_img_path)
        dx = -dx  # the stage has TR origin, OpenCV has TL origin

        bad_measurement = conf < cfg.drift.min_confidence
        if bad_measurement:
            logger.warning(
                f"Low confidence {conf:.3f} for last DIC measurement, skipping correction and hysteresis update for this task"
            )
        # @TODO: do actions when having bad measurements for each task

        task.execute()

        if task.task_type == TaskType.UPDATE_REFERENCE:
            if ref_img_path is None:
                logger.info("No existing reference image")
                exit(1)

            h_manager.update_history(
                x_task=x_task,
                y_task=y_task,
                ref_x=ref_x,
                ref_y=ref_y,
                dx_h=dx,
                dy_h=dy,
                conf_h=conf,
            )

            # Update reference image if needed
            logger.info(
                f"Updated reference image from {ref_img_path} to {current_img_scan_path}"
            )
            ref_img_path = current_img_scan_path

            ref_x = ref_x - dx
            ref_y = ref_y - dy

            logger.info(f"New reference position x={ref_x} y={ref_y}")
        elif task.task_type == TaskType.CORRECTION:
            drift.x.value += dx
            drift.y.value += dy
            logger.info(
                f"Drift relative to actual stage position: dx={dx}, dy={dy} with confidence {conf:.3f}"
            )

            h_manager.update_history(
                x_task=x_task,
                y_task=y_task,
                ref_x=ref_x,
                ref_y=ref_y,
                dx_h=dx,
                dy_h=dy,
                conf_h=conf,
            )
        elif task.task_type == TaskType.EXPOSURE:
            logger.info(
                f"Exposure scan completed at x={x_task} y={y_task} "
                f"drift applied: dx={dx}, dy={dy} with confidence {conf:.3f}"
            )

            h_manager.update_history(
                x_task=x_task,
                y_task=y_task,
                ref_x=ref_x,
                ref_y=ref_y,
                dx_h=dx,
                dy_h=dy,
                conf_h=conf,
            )
        logger.info(f"Accumulated offset: {drift}")
