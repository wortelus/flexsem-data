# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
import math
import time
from math import cos, pi, sin

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMSetScanSetting
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Distance, Rotation

from corrector.config.models import Settings
from corrector.core.coordinates import StagePosition
from corrector.core.dic import dic_measure
from corrector.core.measurement import (
    ReferenceFrame,
    dic_to_stage_offset,
    position_from_dic,
)
from corrector.core.positioning import PositioningError
from corrector.core.scan import scan
from corrector.hysteresis.manager import HysteresisManager
from corrector.utils.sem import send_command_reliable_cfg

logger = logging.getLogger(__name__)


def warmup_spiral(
    seq_length: int,
    center_x: Distance,
    center_y: Distance,
    radius_start: Distance = Distance.from_nanometers(2000),
    radius_end: Distance = Distance.from_nanometers(3000),
    radius_random: Distance = Distance.from_nanometers(1000),
) -> list[tuple[Distance, Distance]]:
    points = []

    for i in range(seq_length):
        angle = (i / seq_length) * 4 * pi
        radius_nm = radius_start.nanometers + (
            (radius_end.nanometers - radius_start.nanometers) * (i / (seq_length - 1))
        )
        x = center_x.nanometers + radius_nm * cos(angle)
        y = center_y.nanometers + radius_nm * sin(angle)
        points.append(
            (
                Distance.from_nanometers(int(x)),
                Distance.from_nanometers(int(y)),
            )
        )

    return points


def warmup_linear(
    seq_length: int,
    warmup_end: tuple[Distance, Distance],
    distance: Distance = Distance.from_nanometers(3000),
) -> list[tuple[Distance, Distance]]:
    end_x, end_y = warmup_end

    if seq_length <= 1:
        return [(end_x, end_y)]

    points = []
    for i in range(seq_length):
        fraction = i / (seq_length - 1)
        x_nm = end_x.nanometers + distance.nanometers * (1 - fraction)
        y_nm = end_y.nanometers - distance.nanometers * (1 - fraction)
        points.append(
            (
                Distance.from_nanometers(int(x_nm)),
                Distance.from_nanometers(int(y_nm)),
            )
        )

    return points


def run_warmup(
    cfg: Settings,
    manager: CommunicationManager,
    cmd_factory: MessageFactory,
    h_mgr: HysteresisManager,
    points: list[tuple[Distance, Distance]],
    site_id: str,
) -> ReferenceFrame:
    """Seed hysteresis history and return a DIC-derived final image anchor."""

    if not points:
        raise PositioningError("Warmup trajectory is empty")

    logger.info("Starting warmup sequence execution")

    def send_warmup_command(command) -> None:
        try:
            send_command_reliable_cfg(
                command,
                cfg=cfg,
                cmd_factory=cmd_factory,
                manager=manager,
            )
        except Exception as exc:
            raise PositioningError(f"Warmup stage command failed: {exc}") from exc

    p0_x, p0_y = points[0]
    if len(points) > 1:
        p1_x, p1_y = points[1]
        sign_x = 1 if (p0_x - p1_x).nanometers >= 0 else -1
        sign_y = 1 if (p0_y - p1_y).nanometers >= 0 else -1
    else:
        sign_x = 1
        sign_y = 1

    # Large unmeasured step used only to prepare the mechanism.
    large_step_x = p0_x + Distance.from_nanometers(sign_x * 3000)
    large_step_y = p0_y + Distance.from_nanometers(sign_y * 3000)
    large_step_command = cmd_factory.set_stage_xyr(
        large_step_x,
        large_step_y,
        Rotation.from_degrees(0),
    )
    send_warmup_command(large_step_command)

    initial_position = StagePosition(x=p0_x, y=p0_y)
    initial_command = cmd_factory.set_stage_xyr(
        initial_position.x,
        initial_position.y,
        Rotation.from_degrees(0),
    )
    send_warmup_command(initial_command)

    set_unfreeze = cmd_factory.set_scan_setting(SEMSetScanSetting.Run)
    send_warmup_command(set_unfreeze)
    set_magnification = cmd_factory.set_magnification(cfg.drift.correction_mag)
    send_warmup_command(set_magnification)
    time.sleep(5)

    scan_type = cfg.drift.correction_type
    try:
        reference_image_path, reference_metadata_path = scan(
            cfg,
            cmd_factory,
            manager,
            scan_type,
            "warmup_ref",
        )
    except Exception as exc:
        raise PositioningError(f"Warmup reference scan failed: {exc}") from exc
    source_reference = ReferenceFrame(
        site_id=site_id,
        position=initial_position,
        image_path=reference_image_path,
        metadata_path=reference_metadata_path,
    )

    final_reference = source_reference
    for index, (command_x, command_y) in enumerate(points, start=1):
        motor_command = StagePosition(x=command_x, y=command_y)
        logger.info("Warmup step %d/%d", index, len(points))

        set_stage_xy = cmd_factory.set_stage_xyr(
            motor_command.x,
            motor_command.y,
            Rotation.from_degrees(0),
        )
        send_warmup_command(set_stage_xy)
        time.sleep(3)

        try:
            image_path, metadata_path = scan(
                cfg,
                cmd_factory,
                manager,
                scan_type,
                "warmup",
            )
            (image_dx, image_dy), confidence = dic_measure(
                cfg,
                source_reference.image_path,
                image_path,
                metadata_path,
                override_mask_disable=True,
            )
            if not math.isfinite(confidence) or confidence < cfg.drift.min_confidence:
                raise PositioningError(
                    f"Warmup DIC confidence {confidence!r} is below required "
                    f"{cfg.drift.min_confidence} at step {index}"
                )

            stage_offset = dic_to_stage_offset(image_dx, image_dy)
            observed_position = position_from_dic(source_reference, stage_offset)
        except PositioningError:
            raise
        except Exception as exc:
            raise PositioningError(
                f"Warmup scan or DIC failed at step {index}: {exc}"
            ) from exc
        h_mgr.record_movement(
            command=motor_command,
            observed_position=observed_position,
            confidence=confidence,
        )
        final_reference = ReferenceFrame(
            site_id=site_id,
            position=observed_position,
            image_path=image_path,
            metadata_path=metadata_path,
        )
        logger.info(
            "Warmup DIC observation: image_dx=%s image_dy=%s, "
            "observed_x=%s observed_y=%s, confidence=%.3f",
            image_dx,
            image_dy,
            observed_position.x,
            observed_position.y,
            confidence,
        )

    logger.info("Warmup sequence execution completed")
    return final_reference
