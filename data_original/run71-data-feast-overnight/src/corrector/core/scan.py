# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
import shutil
import time
from pathlib import Path

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMScanMode, SEMSetScanSetting
from flexsem.state.manager import CommunicationManager

from corrector.config.models import ExposureStepConfig
from src.corrector.config.models import Settings

logger = logging.getLogger(__name__)


def scan(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    scan_type: SEMScanMode,
    save_suffix: str,
) -> (Path, Path):
    # Unfreeze
    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    manager.send_command_reliable(set_unfreeze, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Scan unfrozen for {scan_type.name}")

    # Set scan mode
    set_scan_mode = cmd_factory.set_scan_mode(scan_type)
    manager.send_command_reliable(set_scan_mode, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Exposure mode set to {set_scan_mode.to_wire_format()}")

    # Perform SCAN
    set_direct_save = cmd_factory.set_direct_save(save_all=False)
    manager.send_command_reliable(set_direct_save, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Direct save set to {set_direct_save.to_wire_format()}")

    # Unfreeze
    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    manager.send_command_reliable(set_unfreeze, timeout=cfg.misc.command_timeout_ms)
    logger.debug(f"Scan unfrozen for {scan_type.name}")

    # Move & rename to local temp directory ('e' for exposure)
    src_img_filepath = (
        cfg.misc.sem_scan_path.with_suffix(f".{cfg.misc.sem_scan_image_filetype}")
    )
    src_metadata_filepath = cfg.misc.sem_scan_path.with_suffix(".txt")
    t = time.time()
    temp_img_filepath = (
        cfg.misc.temp_path / f"{save_suffix}_{t}.{cfg.misc.sem_scan_image_filetype}"
    )
    temp_metadata_filepath = cfg.misc.temp_path / f"{save_suffix}_{t}.txt"

    # @TODO: remove ?
    # time.sleep(1)

    # Shutil move operation
    shutil.move(src_img_filepath, temp_img_filepath)
    shutil.move(src_metadata_filepath, temp_metadata_filepath)

    return temp_img_filepath, temp_metadata_filepath


def correction_scan(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    settle_time_s: float = 0.0,
) -> (str, str):

    # Unfreeze
    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    manager.send_command_reliable(set_unfreeze, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Correction scan started, unfrozen")

    # Set magnification
    set_magnification = cmd_factory.set_magnification(cfg.drift.correction_mag)
    manager.send_command_reliable(
        set_magnification, timeout=cfg.misc.command_timeout_ms
    )
    logger.debug(f"Magnification set to {set_magnification.to_wire_format()}")

    # Settle time
    if settle_time_s > 0:
        logger.debug(f"Waiting for settle time: {settle_time_s}s")
        time.sleep(settle_time_s)

    return scan(
        cfg, cmd_factory, manager, cfg.drift.correction_type, "c"
    )


def exposure_scan(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    step: ExposureStepConfig,
) -> (Path, Path):
    # Use override if set, otherwise default
    mag = step.exposure_mag or cfg.drift.exposure_mag
    # @TODO: per step config
    exp_type = cfg.exposure.exposure_type

    # Unfreeze
    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    manager.send_command_reliable(set_unfreeze, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Exposure scan started, unfrozen")

    # Set magnification
    set_magnification = cmd_factory.set_magnification(mag)
    manager.send_command_reliable(
        set_magnification, timeout=cfg.misc.command_timeout_ms
    )
    logger.debug(f"Magnification set to {set_magnification.to_wire_format()}")

    return scan(cfg, cmd_factory, manager, exp_type, "e")
