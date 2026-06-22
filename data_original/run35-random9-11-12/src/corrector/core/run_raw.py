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
from corrector.config.models import Settings, FocusConfig
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


    set_mag = cmd_factory.set_magnification(cfg.drift.correction_mag)
    manager.send_command_blocking(set_mag, timeout=cfg.misc.command_timeout_ms)
    time.sleep(2)  # Wait for the mag to stabilize

    # Set focus
    mag = cfg.drift.correction_mag
    correction_focus: FocusConfig = cfg.focus[mag]
    coarse, fine = correction_focus.coarse, correction_focus.fine
    set_focus = cmd_factory.set_focus(coarse=coarse, fine=fine)
    manager.send_command_blocking(set_focus, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Focus set to {set_focus.to_wire_format()}")


    set_stage_xyr = cmd_factory.set_stage_xyr(x=Distance(0), y=Distance(0), r=Rotation(0))
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    time.sleep(2)  # Wait for the stage to stabilize


    num = 1000
    for i in range(num):
        logger.info(f"Executing task {i + 1}/{num}")

        scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=f"raw_drift_{i}")

        logger.info(f"Saved raw drift image {i + 1}/{num} at time {time.strftime('%Y-%m-%d %H:%M:%S')}")

        time.sleep(300) # Wait for 5 minutes before the next scan