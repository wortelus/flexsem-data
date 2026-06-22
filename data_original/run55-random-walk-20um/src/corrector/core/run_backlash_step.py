# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import json
import logging
import time
from math import cos, sin

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMScanMode
from flexsem.config.models import Settings as FlexSEMSettings
from flexsem.state.manager import CommunicationManager
from flexsem.utils.metric import Distance, Rotation

from corrector.config.models import Settings, FocusConfig
from corrector.config.sanity import sanity
from corrector.core.drift import calculate_drift
from corrector.core.scan import scan
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
            position = {
                "x": Distance.from_nanometers(int(pos_data[0])),
                "y": Distance.from_nanometers(int(pos_data[1])),
                "z": Distance.from_nanometers(int(pos_data[2])),
                "t": Distance.from_nanometers(int(pos_data[3])),
                "r": Rotation.from_degrees(int(pos_data[4])),
            }

            logger.info(f"check r: {position['r'].degrees}")
            assert position['r'] == Rotation.from_degrees(0.0)

            return position
        else:
            logger.error("Failed to fetch position, the response contains no data")
            return None
    except Exception as e:
        logger.error(f"Error while fetching position: {e}")
        return None


R_ZERO = Rotation.from_degrees(0.0)


def measure_backlash(cfg: Settings, cmd_factory: MessageFactory, manager: CommunicationManager,
                     x: Distance, y: Distance, r_angle_move: Rotation,
                     move: Distance,
                     iteration: int, steps: int):
    # placeholder
    dx, dy, conf = Distance.from_nanometers(0), Distance.from_nanometers(0), 0.0

    angle_rad = r_angle_move.radians
    x_move_angled = Distance.from_nanometers(
        int(move.nanometers * cos(angle_rad))
    )
    y_move_angled = Distance.from_nanometers(
        int(move.nanometers * sin(angle_rad))
    )

    x_move_angled_total = Distance.from_nanometers(
        int(x_move_angled.nanometers * steps)
    )
    y_move_angled_total = Distance.from_nanometers(
        int(y_move_angled.nanometers * steps)
    )

    logger.info(
        f"Moving for backlash measure: x={(x + x_move_angled).nanometers}/{x.nanometers}, y={(y + y_move_angled).nanometers}/{y.nanometers} for angle {r_angle_move.degrees}deg")

    set_mag = cmd_factory.set_magnification(cfg.drift.correction_mag)
    manager.send_command_blocking(set_mag, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Magnification set to {cfg.drift.correction_mag}")

    # Set focus
    # mag = cfg.drift.correction_mag
    # correction_focus: FocusConfig = cfg.focus[mag]
    # coarse, fine = correction_focus.coarse, correction_focus.fine
    # set_focus = cmd_factory.set_focus(coarse=coarse, fine=fine)
    # manager.send_command_blocking(set_focus, timeout=cfg.misc.command_timeout_ms)
    # logger.info(f"Focus set to mag {mag} with coarse={coarse}, fine={fine}")

    # backlash position
    set_stage_xyr = cmd_factory.set_stage_xyr(x + x_move_angled_total, y + y_move_angled_total, R_ZERO)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    logger.info(
        f"Moved to backlash position x={(x + x_move_angled_total).nanometers}, y={(y + y_move_angled_total).nanometers}")
    time.sleep(1)

    # home move
    set_stage_xyr = cmd_factory.set_stage_xyr(x, y, R_ZERO)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Moved back to home position x={x.nanometers}, y={y.nanometers}")
    time.sleep(1)

    # SCAN reference
    ref_img_path, ref_meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1,
                                       save_suffix=f"{iteration:05d}_a_reference")

    sub_history = []

    # forward move
    for step in range(steps):
        x_move_angled_step = Distance.from_nanometers(
            int(x_move_angled.nanometers * (step + 1))
        )
        y_move_angled_step = Distance.from_nanometers(
            int(y_move_angled.nanometers * (step + 1))
        )

        # backlash move
        set_stage_xyr = cmd_factory.set_stage_xyr(x + x_move_angled_step, y + y_move_angled_step, R_ZERO)
        manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
        logger.info(f"Moved to x={(x + x_move_angled_step).nanometers}, y={(y + y_move_angled_step).nanometers}")

        time.sleep(1)  # Wait for the stage to settle

        # SCAN
        filename = f"{iteration:05d}_b_forward_{step:03d}"
        logger.info(
            f"Saving {filename} of x={(x + x_move_angled_step).nanometers}, y={(y + y_move_angled_step).nanometers}")
        img_path, meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=filename)

        # Measure
        # metadata = load_metadata(meta_path)
        # logger.debug(f"Reference image for drift calculation: {ref_img_path}")
        # logger.debug(f"Loaded metadata for drift calculation: {metadata} from {meta_path}")
        # # dx, dy is of type Distance
        # (dx, dy), conf = calculate_drift(cfg, img_path, ref_img_path, metadata)
        # # we need only to flip the x
        # dx = -dx
        # logger.info(f"Drift detected at (x={dx}, y={dy}) with conf={conf} to the original reference image")

        sub_history.append({
            "step": step,
            "r_target": r_angle_move.degrees,
            "x_target": x_move_angled.nanometers,
            "y_target": y_move_angled.nanometers,
            "x_move": x_move_angled_step.nanometers,
            "y_move": y_move_angled_step.nanometers,
            "dx_measured": dx.nanometers,
            "dy_measured": dy.nanometers,
            "conf": f"{conf:.4f}",
            "img_path": img_path.as_posix(),
            "meta_path": meta_path.as_posix(),
        })

    for i in range(steps):
        x_move_angled_step = Distance.from_nanometers(
            int(x_move_angled.nanometers * (steps - i - 1))
        )
        y_move_angled_step = Distance.from_nanometers(
            int(y_move_angled.nanometers * (steps - i - 1))
        )

        # backlash move
        set_stage_xyr = cmd_factory.set_stage_xyr(x + x_move_angled_step, y + y_move_angled_step, R_ZERO)
        manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
        logger.info(f"Moved to x={(x + x_move_angled_step).nanometers}, y={(y + y_move_angled_step).nanometers}")

        time.sleep(1)  # Wait for the stage to settle

        # SCAN
        filename = f"{iteration:05d}_c_backward_{i:03d}"
        logger.info(
            f"Saving {filename} of x={(x + x_move_angled_step).nanometers}, y={(y + y_move_angled_step).nanometers}")
        img_path, meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=filename)

        # Measure
        # metadata = load_metadata(meta_path)
        # logger.debug(f"Reference image for drift calculation: {ref_img_path}")
        # logger.debug(f"Loaded metadata for drift calculation: {metadata} from {meta_path}")
        # # dx, dy is of type Distance
        # (dx, dy), conf = calculate_drift(cfg, img_path, ref_img_path, metadata)
        # # we need only to flip the x
        # dx = -dx
        # logger.info(f"Drift detected at (x={dx}, y={dy}) with conf={conf} to the original reference image")

        sub_history.append({
            "step": steps + i,
            "r_target": r_angle_move.degrees,
            "x_target": x_move_angled.nanometers,
            "y_target": y_move_angled.nanometers,
            "x_move": x_move_angled_step.nanometers,
            "y_move": y_move_angled_step.nanometers,
            "dx_measured": dx.nanometers,
            "dy_measured": dy.nanometers,
            "conf": f"{conf:.4f}",
            "img_path": img_path.as_posix(),
            "meta_path": meta_path.as_posix(),
        })

    # returning (home) move
    set_stage_xyr = cmd_factory.set_stage_xyr(x, y, R_ZERO)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Returned to home position x={x.nanometers}, y={y.nanometers}")

    time.sleep(1)  # Wait for the stage to settle

    # SCAN
    img_path, meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=f"{iteration:05d}_d_final")

    metadata = load_metadata(meta_path)
    logger.debug(f"Reference image for drift calculation: {ref_img_path}")
    logger.debug(f"Loaded metadata for drift calculation: {metadata} from {meta_path}")
    # dx, dy is of type Distance
    (dx, dy), conf = calculate_drift(cfg, img_path, ref_img_path, metadata)
    # we need only to flip the x
    dx = -dx
    logger.info(f"Drift detected at (x={dx}, y={dy}) with conf={conf} to the original reference image")

    sub_history.append({
        "step": "final",
        "r_target": r_angle_move.degrees,
        "x_target": x.nanometers,
        "y_target": y.nanometers,
        "x_move": 0,
        "y_move": 0,
        "dx_measured": dx.nanometers,
        "dy_measured": dy.nanometers,
        "conf": f"{conf:.4f}",
        "img_path": img_path.as_posix(),
        "meta_path": meta_path.as_posix(),
    })

    return sub_history


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

    # posun po x "doleva", postupný posuv po Y
    tasks = [
        (Distance.from_millimeters(0),
         Distance.from_millimeters(-0.9),
         Rotation.from_degrees(0.0))
    ]
    steps_down_y = [Distance.from_millimeters(-0.8),
                    Distance.from_millimeters(-0.6),
                    Distance.from_millimeters(-0.2),
                    Distance.from_millimeters(0.4),
                    Distance.from_millimeters(1.2)]
    for y in steps_down_y:
        tasks.append((Distance.from_millimeters(0), y, Rotation.from_degrees(0.0)))

    # posun po y "dolů", postupný posuv po X
    tasks.append((
        Distance.from_millimeters(0.9),
        Distance.from_millimeters(0),
        Rotation.from_degrees(90.0)
    ))
    steps_right_x = [Distance.from_millimeters(0.8),
                     Distance.from_millimeters(0.6),
                     Distance.from_millimeters(0.2),
                     Distance.from_millimeters(-0.4),
                     Distance.from_millimeters(-1.2)]
    for x in steps_right_x:
        tasks.append((x, Distance.from_millimeters(0), Rotation.from_degrees(90.0)))

    tasks_num = len(tasks)
    logger.info(f"Total number of tasks to execute: {tasks_num}")

    move = Distance.from_nanometers(100)
    steps = 50

    bl = Distance.from_nanometers(1000)
    cancel_backlash = cmd_factory.backlash_cancel(bl)
    manager.send_command_blocking(cancel_backlash, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Backlash compensation command sent of {bl.nanometers} nm")

    history_filename = time.strftime("backlash_history_%Y%m%d_%H%M%S.json")
    logger.info(f"Backlash measurement history will be saved to {history_filename}")

    with open(history_filename, 'w') as f:
        history = []
        for i, task in enumerate(tasks):
            x_task, y_task, r_task = task
            logger.info(f"Executing task {i + 1}/{len(tasks)}")

            logger.info(f"Backlash compensation command sent of {bl.nanometers} nm")
            sub_history = measure_backlash(cfg, cmd_factory, manager,
                                           x_task, y_task, r_task,
                                           move,
                                           i + 1, steps=steps)

            experiment = {
                "task_id": i + 1,
                "total_tasks": tasks_num,
                "x_target": x_task.nanometers,
                "y_target": y_task.nanometers,
                "r_target": r_task.degrees,
                "move": move.nanometers,
                "steps": steps,
                "sub_history": sub_history
            }
            history.append(experiment)

            f.write(f"{json.dumps(experiment, indent=2)}\n")
            f.flush()
            logger.info(f"Experiment data saved to {history_filename}")
    logger.info(f"Backlash measurement completed. History saved to {history_filename}")
    logger.info("All tasks executed, exiting.")
    manager.disconnect()
