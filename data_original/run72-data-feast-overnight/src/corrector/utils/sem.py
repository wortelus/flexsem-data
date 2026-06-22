# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from flexsem.commands import MessageFactory
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Distance, Rotation
from flexsem.utils.exception import ReliableCommandError

from corrector.config.models import Settings

import logging

logger = logging.getLogger(__name__)


def get_current_position(cfg: Settings,
                         manager: CommunicationManager,
                         cmd_factory: MessageFactory) -> dict | None:
    """
    Fetches and returns the current stage position (X, Y, Z, T, R) in nanometers/microradians.
    """
    try:
        logger.debug("Fetching current stage position (get_stage_xyztr)...")
        msg = cmd_factory.get_stage_xyztr()
        response = send_command_reliable_cfg(msg, cfg=cfg, cmd_factory=cmd_factory, manager=manager)

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


def send_command_reliable_cfg(command,
                              cfg: Settings,
                              cmd_factory: MessageFactory,
                              manager: CommunicationManager):
    try:
        return manager.send_command_reliable(command,
                                             timeout=cfg.misc.command_timeout_s,
                                             max_retries=cfg.misc.max_retries,
                                             retry_delay=cfg.misc.retry_delay_s,
                                             reconnect_delay=cfg.misc.reconnect_delay_s,
                                             max_reconnect_retries=cfg.misc.max_reconnect_retries)
    except ReliableCommandError as e:
        logger.error(f"Command failed after {e.attempts} attempts: {e}")
        if cfg.misc.turn_beam_off_on_error:
            logger.warning(f"Turning beam off due to command failure, the SEM refused to execute the command after "
                           f"{cfg.misc.max_retries} retries and "
                           f"{cfg.misc.max_reconnect_retries} reconnect attempts.")
            try:
                beam_off_cmd = cmd_factory.set_beam_state("OFF")
                manager.send_command_reliable(beam_off_cmd,
                                              timeout=cfg.misc.command_timeout_s,
                                              max_retries=cfg.misc.max_retries,
                                              retry_delay=cfg.misc.retry_delay_s,
                                              reconnect_delay=cfg.misc.reconnect_delay_s,
                                              max_reconnect_retries=cfg.misc.max_reconnect_retries)
                logger.info("Beam turned off successfully.")

                # Exit with error code after attempting to turn beam off
                exit(1)
            except ReliableCommandError as e2:
                logger.error(f"Failed to turn beam off after command failure: {e2}, attempted to send beam off command but it also failed after {e2.attempts} attempts.")

                # Exit with error code even if turning beam off also fails, as the system is in an unknown state and it's safer to exit than to continue operating
                exit(1)
