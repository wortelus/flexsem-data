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

from corrector.config.models import Settings
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


def measure_backlash(cfg: Settings, cmd_factory: MessageFactory, manager: CommunicationManager,
                     x: Distance, y: Distance, r_angle_move: Rotation,
                     move: Distance,
                     i: int):
    r_zero = Rotation.from_degrees(0)

    angle_rad = r_angle_move.radians
    x_move_angled = Distance.from_nanometers(
        int(move.nanometers * cos(angle_rad))
    )
    y_move_angled = Distance.from_nanometers(
        int(move.nanometers * sin(angle_rad))
    )

    logger.info(
        f"Moving for backlash measure: x={(x + x_move_angled).nanometers}/{x.nanometers}, y={(y + y_move_angled).nanometers}/{y.nanometers}, r={r_angle_move.degrees}deg for stage angle {r_zero.degrees}deg")

    # backlash position
    set_stage_xyr = cmd_factory.set_stage_xyr(x + x_move_angled, y + y_move_angled, r_zero)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    time.sleep(1)

    # home move
    set_stage_xyr = cmd_factory.set_stage_xyr(x, y, r_zero)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    time.sleep(2)

    # SCAN
    ref_img_path, ref_meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1,
                                       save_suffix=f"measure_backlash_ref_{i}")

    # backlash move
    set_stage_xyr = cmd_factory.set_stage_xyr(x + x_move_angled, y + y_move_angled, r_zero)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    time.sleep(1)  # Wait for the stage to settle

    # returning (home) move
    set_stage_xyr = cmd_factory.set_stage_xyr(x, y, r_zero)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    time.sleep(2)  # Wait for the stage to settle

    # SCAN
    img_path, meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=f"measure_backlash_{i}")

    metadata = load_metadata(meta_path)
    logger.debug(f"Reference image for drift calculation: {ref_img_path}")
    logger.debug(f"Loaded metadata for drift calculation: {metadata} from {meta_path}")
    (dx, dy), conf = calculate_drift(cfg, img_path, ref_img_path, metadata)
    dx = -dx
    logger.info(f"Drift detected at (x={dx}, y={dy}) with conf={conf} to the original reference image")

    return dx, dy, img_path, meta_path, conf


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

    angles = [Rotation.from_degrees(deg) for deg in [0.0, 45.0, 90, 135]]
    x_nm = [Distance.from_millimeters(mm) for mm in [-1.5, 0, 1.5]]
    y_nm = [Distance.from_millimeters(mm) for mm in [-1.5, 0, 1.5]]

    tasks = []
    for r in angles:
        for x in x_nm:
            for y in y_nm:
                tasks.append((x, y, r))

    tasks_num = len(angles) * len(x_nm) * len(y_nm)
    logger.info(f"Total number of tasks to execute: {tasks_num}")

    move = Distance.from_nanometers(1500)

    history_filename = time.strftime("backlash_history_%Y%m%d_%H%M%S.json")
    logger.info(f"Backlash measurement history will be saved to {history_filename}")


    with open(history_filename, 'w') as f:

        history = []
        for i, task in enumerate(tasks):
            x_task, y_task, r_task = task
            logger.info(f"Executing task {i + 1}/{len(tasks)}")

            dx, dy, img_path, meta_path, conf = measure_backlash(cfg, cmd_factory, manager,
                             x_task, y_task, r_task,
                             move,
                             i + 1)

            experiment = {
                "task": i + 1,
                "x_target": x_task.nanometers,
                "y_target": y_task.nanometers,
                "r_target": r_task.degrees,
                "move": move.nanometers,
                "dx_measured": dx.nanometers,
                "dy_measured": dy.nanometers,
                "conf": f"{conf:.4f}",
                "img_path": img_path.as_posix(),
                "meta_path": meta_path.as_posix(),
            }

            history.append(experiment)

            f.write(f"{json.dumps(experiment)}\n")
            f.flush()
            logger.info(f"Experiment data saved to {history_filename}")
    logger.info(f"Backlash measurement completed. History saved to {history_filename}")
    logger.info("All tasks executed, exiting.")
    manager.disconnect()

