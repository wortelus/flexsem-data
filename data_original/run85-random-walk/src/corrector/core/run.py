# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

"""Collect SEM scans along a bounded random walk for offline DIC processing."""

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

from flexsem.commands import MessageFactory
from flexsem.commands.enum import SEMScanMode
from flexsem.config.models import Settings as FlexSEMSettings
from flexsem.state.manager import CommunicationManager
from flexsem.utils import Distance, Rotation

from corrector.config.models import Settings
from corrector.config.sanity import sanity
from corrector.core.scan import scan
from corrector.utils.sem import get_current_position, send_command_reliable_cfg

logger = logging.getLogger(__name__)

R_ZERO = Rotation.from_degrees(0.0)
SETTLE_TIME_S = 1.0
MAGNIFICATION = 8000
SCAN_MODE = SEMScanMode.Slow1
RANDOM_WALK_STEPS = 20000
WALK_RADIUS_NM = 20_000
MAX_DIRECTION_ATTEMPTS = 1000

# Each tuple is (minimum step, maximum step, sampling weight).  The resulting
# acquisition emphasizes fine positioning while retaining mixed coarse/fine
# histories for the inverse hysteresis model.
STEP_SIZE_BANDS_NM = (
    (50, 100, 0.05),
    (100, 500, 0.20),
    (500, 1_000, 0.20),
    (1_000, 2_500, 0.20),
    (2_500, 5_000, 0.15),
    (5_000, 10_000, 0.10),
    (10_000, 15_000, 0.07),
    (15_000, 20_000, 0.03),
)


def _validate_walk_parameters(
    radius_nm: int,
    step_size_bands: tuple[tuple[int, int, float], ...],
) -> None:
    if radius_nm <= 0:
        raise ValueError("Random-walk radius must be positive")
    if not step_size_bands:
        raise ValueError("At least one step-size band is required")

    for minimum_nm, maximum_nm, weight in step_size_bands:
        if minimum_nm <= 0 or maximum_nm < minimum_nm:
            raise ValueError(
                f"Invalid step-size band ({minimum_nm}, {maximum_nm})"
            )
        if weight <= 0:
            raise ValueError("Step-size band weights must be positive")
        if maximum_nm > radius_nm:
            raise ValueError("A step cannot exceed the bounded-walk radius")


def generate_bounded_random_walk_2d(
    base_position: dict,
    num_steps: int,
    radius: Distance,
    rng: random.Random,
    step_size_bands: tuple[tuple[int, int, float], ...] = STEP_SIZE_BANDS_NM,
) -> list[tuple[Distance, Distance]]:
    """Generate consecutive random steps constrained to a circle around base."""

    if num_steps < 0:
        raise ValueError("Number of random-walk steps must be non-negative")

    radius_nm = radius.nanometers
    _validate_walk_parameters(radius_nm, step_size_bands)

    base_x_nm = base_position["x"].nanometers
    base_y_nm = base_position["y"].nanometers
    current_x_nm = 0
    current_y_nm = 0
    bands = [(minimum_nm, maximum_nm) for minimum_nm, maximum_nm, _ in step_size_bands]
    weights = [weight for _, _, weight in step_size_bands]
    points = []

    for step_number in range(1, num_steps + 1):
        minimum_nm, maximum_nm = rng.choices(bands, weights=weights, k=1)[0]
        step_size_nm = rng.uniform(minimum_nm, maximum_nm)

        for _ in range(MAX_DIRECTION_ATTEMPTS):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            candidate_x_nm = current_x_nm + int(round(step_size_nm * math.cos(angle)))
            candidate_y_nm = current_y_nm + int(round(step_size_nm * math.sin(angle)))
            if math.hypot(candidate_x_nm, candidate_y_nm) <= radius_nm:
                current_x_nm = candidate_x_nm
                current_y_nm = candidate_y_nm
                break
        else:
            raise RuntimeError(
                f"Could not place random-walk step {step_number} inside "
                f"the {radius_nm}nm radius after {MAX_DIRECTION_ATTEMPTS} attempts"
            )

        points.append(
            (
                Distance.from_nanometers(base_x_nm + current_x_nm),
                Distance.from_nanometers(base_y_nm + current_y_nm),
            )
        )

    return points


def _create_history_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_filename = Path(f"hysteresis_dataset_{timestamp}.jsonl")
    with history_filename.open("x", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return history_filename


def _append_record(history_filename: Path, record: dict) -> None:
    """Persist one complete JSONL record before the next SEM operation."""

    with history_filename.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _placeholder_record(
    experiment_name: str,
    iteration: int,
    step: int,
    target_x: Distance,
    target_y: Distance,
    image_path: Path,
) -> dict:
    """Keep the existing JSON schema; actual positions are filled in offline."""

    return {
        "timestamp": time.time(),
        "experiment_name": experiment_name,
        "iteration": iteration,
        "step": step,
        "x_target_abs": target_x.nanometers,
        "y_target_abs": target_y.nanometers,
        "x_actual_abs": 0,
        "y_actual_abs": 0,
        "confidence": "0.0000",
        "img_path": image_path.as_posix(),
    }


def execute_and_record_trajectory(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    trajectory_points: list[tuple[Distance, Distance]],
    base_position: dict,
    experiment_name: str,
    iteration: int,
    history_filename: Path,
) -> None:
    """Execute one trajectory and durably append each saved scan to JSONL."""

    set_magnification = cmd_factory.set_magnification(MAGNIFICATION)
    send_command_reliable_cfg(
        set_magnification,
        cfg=cfg,
        cmd_factory=cmd_factory,
        manager=manager,
    )

    base_x = base_position["x"]
    base_y = base_position["y"]
    move_to_base = cmd_factory.set_stage_xyr(base_x, base_y, R_ZERO)
    send_command_reliable_cfg(
        move_to_base,
        cfg=cfg,
        cmd_factory=cmd_factory,
        manager=manager,
    )
    logger.info(
        "Moved to random-walk base motor command x=%snm y=%snm",
        base_x.nanometers,
        base_y.nanometers,
    )
    time.sleep(SETTLE_TIME_S)

    reference_image_path, _ = scan(
        cfg,
        cmd_factory,
        manager,
        SCAN_MODE,
        save_suffix=f"{iteration:05d}_{experiment_name}_000_reference",
    )
    _append_record(
        history_filename,
        _placeholder_record(
            experiment_name,
            iteration,
            0,
            base_x,
            base_y,
            reference_image_path,
        ),
    )

    for step, (target_x, target_y) in enumerate(trajectory_points, start=1):
        move_command = cmd_factory.set_stage_xyr(target_x, target_y, R_ZERO)
        send_command_reliable_cfg(
            move_command,
            cfg=cfg,
            cmd_factory=cmd_factory,
            manager=manager,
        )
        logger.info(
            "Random-walk step %d/%d: motor target x=%snm y=%snm",
            step,
            len(trajectory_points),
            target_x.nanometers,
            target_y.nanometers,
        )
        time.sleep(SETTLE_TIME_S)

        image_path, _ = scan(
            cfg,
            cmd_factory,
            manager,
            SCAN_MODE,
            save_suffix=f"{iteration:05d}_{experiment_name}_{step:05d}",
        )
        _append_record(
            history_filename,
            _placeholder_record(
                experiment_name,
                iteration,
                step,
                target_x,
                target_y,
                image_path,
            ),
        )

    return_to_base = cmd_factory.set_stage_xyr(base_x, base_y, R_ZERO)
    send_command_reliable_cfg(
        return_to_base,
        cfg=cfg,
        cmd_factory=cmd_factory,
        manager=manager,
    )
    logger.info("Random walk finished; returning to base")
    time.sleep(SETTLE_TIME_S)

    final_image_path, _ = scan(
        cfg,
        cmd_factory,
        manager,
        SCAN_MODE,
        save_suffix=f"{iteration:05d}_{experiment_name}_final",
    )
    _append_record(
        history_filename,
        _placeholder_record(
            experiment_name,
            iteration,
            len(trajectory_points) + 1,
            base_x,
            base_y,
            final_image_path,
        ),
    )


def run(cfg: Settings, sem_cfg: FlexSEMSettings) -> None:
    logger.info("Starting bounded random-walk hysteresis data collection")
    sanity(cfg)

    history_filename = _create_history_file()
    logger.info("Dataset is appended durably to %s", history_filename)

    cmd_factory = MessageFactory(
        server_uid=sem_cfg.server.unit_id,
        client_uid=sem_cfg.client.unit_id,
    )
    manager = CommunicationManager(sem_cfg)
    manager.connect()
    logger.info("Connected to %s", sem_cfg.server.unit_id)

    try:
        reported_motor_position = get_current_position(cfg, manager, cmd_factory)
        if reported_motor_position is None:
            raise RuntimeError("Could not fetch the base motor command")

        base_position = {
            "x": reported_motor_position["x"],
            "y": reported_motor_position["y"],
        }
        random_seed = time.time_ns()
        rng = random.Random(random_seed)
        experiment_name = f"bounded_random_walk_seed_{random_seed}"
        logger.info(
            "Random-walk base x=%snm y=%snm, radius=%snm, seed=%s",
            base_position["x"].nanometers,
            base_position["y"].nanometers,
            WALK_RADIUS_NM,
            random_seed,
        )

        trajectory = generate_bounded_random_walk_2d(
            base_position,
            RANDOM_WALK_STEPS,
            Distance.from_nanometers(WALK_RADIUS_NM),
            rng,
        )
        execute_and_record_trajectory(
            cfg,
            cmd_factory,
            manager,
            trajectory,
            base_position,
            experiment_name,
            iteration=1,
            history_filename=history_filename,
        )
    except Exception:
        logger.exception(
            "Data collection stopped; all completed records remain in %s",
            history_filename,
        )
        raise
    finally:
        try:
            beam_off = cmd_factory.set_beam_state("OFF")
            send_command_reliable_cfg(
                beam_off,
                cfg=cfg,
                cmd_factory=cmd_factory,
                manager=manager,
            )
        finally:
            manager.disconnect()
            logger.info("Disconnected; dataset remains at %s", history_filename)
