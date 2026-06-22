# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from time import sleep

from flexsem.commands import MessageFactory
from flexsem.config.models import Settings as FlexSEMSettings
from flexsem.state.manager import CommunicationManager
from flexsem.utils.metric import Distance, Rotation

from corrector.config.metric import XYR
from corrector.config.models import Settings
from corrector.config.sanity import sanity
from corrector.core.drift import calculate_drift
from corrector.core.task import TaskType, build_task_list
from corrector.hysteresis.manager import HysteresisManager
from corrector.hysteresis.warmup import warmup_spiral, run_warmup, warmup_linear
from corrector.misc.metadata import load_metadata

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
            # @TODO: make it its own type or something
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


def update_hysteresis_history(h_manager: HysteresisManager,
                              x_task: Distance,
                              y_task: Distance,
                              ref_x: Distance,
                              ref_y: Distance,
                              dx_h: Distance,
                              dy_h: Distance,
                              conf_h: float):
    logger.info(f"Updating hysteresis movement with (x={dx_h}, y={dy_h}) with conf={conf_h}")

    # Update hysteresis manager history
    h_manager.update_history(x_task.nanometers,
                             y_task.nanometers,
                             ref_x.nanometers - dx_h.nanometers,
                             ref_y.nanometers - dy_h.nanometers)


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

    warmup_end = cfg.exposure.steps[0].xyr
    warmup_end_xy = warmup_end.x.value, warmup_end.y.value

    h_manager = HysteresisManager(cfg.hysteresis)
    warmup = warmup_linear(h_manager.sequence_length + 2,
                           warmup_end_xy,
                           cfg.hysteresis.warmup_point.y.value)
    logger.info(
        f"{len(warmup)} warmup steps generated around x={cfg.hysteresis.warmup_point.x.value} y={cfg.hysteresis.warmup_point.y.value}")

    run_warmup(cfg, manager, cmd_factory, h_manager, warmup)

    tasks = build_task_list(cfg, cmd_factory, manager)
    logger.info(f"Built task list with {len(tasks)} tasks")

    ref_img_path = None
    ref_x, ref_y = None, None
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
        manager.send_command_blocking(
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

        (dx, dy, conf), current_img_scan_path, current_img_metadata_path = task.execute(ref_img_path=ref_img_path)
        dx = -dx  # the stage has TR origin, OpenCV has TL origin

        if task.task_type == TaskType.UPDATE_REFERENCE:
            if ref_img_path is not None:
                if conf >= cfg.drift.min_confidence:
                    sleep(2)  # wait for the stage to settle before scanning the reference image
                    metadata = load_metadata(current_img_metadata_path)
                    logger.info(f"Scanning reference image for hysteresis history update")
                    (dx_ref, dy_ref), conf = calculate_drift(cfg, ref_img_path, current_img_scan_path, metadata, override_mask_disable=True)
                    dx_ref = -dx_ref  # the stage has TR origin, OpenCV has TL origin

                    # @TODO: should we add reference update error to drift value ? maybe two-shot or three-shot the target position ?

                    update_hysteresis_history(
                        h_manager=h_manager,
                        x_task=x_task,
                        y_task=y_task,
                        ref_x=x_task,
                        ref_y=y_task,
                        dx_h=dx_ref,
                        dy_h=dy_ref,
                        conf_h=conf,
                    )
                else:
                    logger.warning(
                        f"Low confidence {conf:.3f} in reference image update, skipping hysteresis history update"
                    )
            else:
                logger.info("No existing reference image, skipping hysteresis history update & setting the first one")

            # Update reference image if needed
            logger.info(
                f"Updated reference image from {ref_img_path} to {current_img_scan_path}, new reference position x={x_actual} y={y_actual} with confidence {conf:.3f}"
            )
            ref_img_path = current_img_scan_path # we don't need to replace the metadata
            ref_x = x_actual
            ref_y = y_actual
        elif task.task_type == TaskType.CORRECTION:
            if conf >= cfg.drift.min_confidence:
                drift.x.value += dx
                drift.y.value += dy
                logger.info(
                    f"Drift relative to actual stage position: dx={dx}, dy={dy} with confidence {conf:.3f}"
                )
            else:
                logger.warning(
                    f"Low confidence {conf:.3f} in drift calculation, skipping drift update"
                )

            update_hysteresis_history(
                h_manager=h_manager,
                x_task=x_task,
                y_task=y_task,
                ref_x=ref_x,
                ref_y=ref_y,
                dx_h=dx,
                dy_h=dy,
                conf_h=conf,
            )

        logger.info(f"Accumulated offset: {drift}")
