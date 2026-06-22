# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

# --- MODIFIED FOR HYSTERESIS DATASET COLLECTION ---

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import json
import logging
import random
import time
from math import cos, sin, pi
from pathlib import Path

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
R_ZERO = Rotation.from_degrees(0.0)
SETTLE_TIME = 1  # seconds

# ==============================================================================
# SECTION 1: TRAJECTORY GENERATION FUNCTIONS
# ==============================================================================
# These functions generate lists of target (x, y) points for the stage to follow.

def generate_sawtooth_trajectory(axis: str, start_pos: dict, full_range: Distance, step_size: Distance,
                                 repetitions: int) -> list:
    """Generates a sawtooth (ramp up, ramp down) trajectory along a single axis."""
    points = []
    current_x, current_y = start_pos['x'], start_pos['y']

    for _ in range(repetitions):
        # Move forward
        pos = Distance.from_nanometers(0)
        while pos.nanometers < full_range.nanometers:
            pos += step_size
            if axis == 'x':
                points.append((current_x + pos, current_y))
            else:
                points.append((current_x, current_y + pos))

        # Move backward
        while pos.nanometers > 0:
            pos -= step_size
            if axis == 'x':
                points.append((current_x + pos, current_y))
            else:
                points.append((current_x, current_y + pos))
    return points


def generate_square_trajectory(first_corner: dict, side_length: Distance, points_per_side: int) -> list:
    """Generates a square trajectory around a center point."""
    points = []
    start_x, start_y = first_corner['x'], first_corner['y']

    assert side_length.nanometers % points_per_side == 0, "Side length must be divisible by points per side"
    side_step = Distance.from_nanometers(int(side_length.nanometers / points_per_side))

    # Top, Right, Bottom, Left
    for i in range(points_per_side):
        side_step_i = Distance.from_nanometers(int(i * side_step.nanometers))
        points.append((
            start_x,
            start_y + side_step_i
        ))
    for i in range(points_per_side):
        side_step_i = Distance.from_nanometers(int(i * side_step.nanometers))
        points.append((
            start_x + side_step_i,
            start_y + side_length
        ))
    for i in range(points_per_side):
        side_step_i = Distance.from_nanometers(int(i * side_step.nanometers))
        points.append((
            start_x + side_length,
            start_y + side_length - side_step_i
        ))
    for i in range(points_per_side):
        side_step_i = Distance.from_nanometers(int(i * side_step.nanometers))
        points.append((
            start_x + side_length - side_step_i,
            start_y))


    return points


def generate_circle_trajectory(center: dict, radius: Distance, num_points: int) -> list:
    """Generates a circular trajectory."""
    points = []
    cx, cy = center['x'], center['y']
    for i in range(num_points + 1):
        angle = 2 * pi * i / num_points
        x = cx + Distance.from_nanometers(int(radius.nanometers * cos(angle)))
        y = cy + Distance.from_nanometers(int(radius.nanometers * sin(angle)))
        points.append((x, y))
    return points


def generate_random_walk_2d(start_pos: dict, num_steps: int, max_step_size: Distance) -> list:
    """Generates a 2D random walk trajectory."""
    points = []
    current_x, current_y = start_pos['x'], start_pos['y']

    points.append((current_x, current_y))  # Include starting position

    for _ in range(num_steps):
        angle = random.uniform(0, 2 * pi)
        # set 50 nm as minimum step as this is the smallest step we can command
        step_dist = random.uniform(50, max_step_size.nanometers)

        current_x = Distance.from_nanometers(int(step_dist * cos(angle)))
        current_y = Distance.from_nanometers(int(step_dist * sin(angle)))
        points.append((start_pos['x'] + current_x, start_pos['y'] + current_y))
    return points


def generate_poi_jumps(points_of_interest: list, num_jumps: int) -> list:
    """Generates random jumps between a predefined set of points."""
    return [random.choice(points_of_interest) for _ in range(num_jumps)]


# ==============================================================================
# SECTION 2: CORE DATA COLLECTION FUNCTION
# ==============================================================================

def execute_and_record_trajectory(
        cfg: Settings,
        cmd_factory: MessageFactory,
        manager: CommunicationManager,
        mag: int,
        trajectory_points: list,
        base_pos: dict,
        experiment_name: str,
        iteration: int,
) -> list:
    """
    Executes a given trajectory, measures actual positions, and records the data.
    """
    history = []

    # 1. Set common imaging parameters (Magnification, Focus)
    set_mag = cmd_factory.set_magnification(mag)
    manager.send_command_blocking(set_mag, timeout=cfg.misc.command_timeout_ms)

    correction_focus: FocusConfig = cfg.focus[mag]
    coarse, fine = correction_focus.coarse, correction_focus.fine
    set_focus = cmd_factory.set_focus(coarse=coarse, fine=fine)
    manager.send_command_blocking(set_focus, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Magnification and focus set for experiment '{experiment_name}'")

    # 2. Move to the starting (base) position of the experiment
    start_x, start_y = base_pos['x'], base_pos['y']
    set_stage_xyr = cmd_factory.set_stage_xyr(start_x, start_y, R_ZERO)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Moved to base position for experiment: x={start_x.nanometers}, y={start_y.nanometers}")
    time.sleep(SETTLE_TIME)  # Use a configurable settle time

    # 3. Take a reference scan at the starting position
    ref_img_path, ref_meta_path = scan(
        cfg, cmd_factory, manager, SEMScanMode.Slow1,
        save_suffix=f"{iteration:05d}_{experiment_name}_000_reference"
    )

    # 3.a Record the reference position
    history.append({
        "timestamp": time.time(),
        "experiment_name": experiment_name,
        "iteration": iteration,
        "step": 0,
        "x_target_abs": start_x.nanometers,
        "y_target_abs": start_y.nanometers,
        "x_actual_abs": start_x.nanometers,
        "y_actual_abs": start_y.nanometers,
        "confidence": "1.0000",
        "img_path": ref_img_path.as_posix(),
    })

    # 4. Iterate through the generated trajectory points
    for step, (target_x, target_y) in enumerate(trajectory_points):

        # --- SEND MOVE COMMAND ---
        set_stage_xyr = cmd_factory.set_stage_xyr(target_x, target_y, R_ZERO)
        manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
        logger.info(
            f"Step {step + 1}/{len(trajectory_points)}: Moving to target x={target_x.nanometers}, y={target_y.nanometers}")
        time.sleep(SETTLE_TIME)

        # --- SCAN & MEASURE ACTUAL POSITION ---
        filename = f"{iteration:05d}_{experiment_name}_{step + 1:03d}"
        img_path, meta_path = scan(cfg, cmd_factory, manager, SEMScanMode.Slow1, save_suffix=filename)

        try:
            metadata = load_metadata(meta_path)
            (dx, dy), conf = calculate_drift(cfg, img_path, ref_img_path, metadata)
            dx = -dx
            logger.info(f"Drift detected: (dx={dx.nanometers}, dy={dy.nanometers}) with conf={conf:.4f}")
            if conf < 0.2:
                logger.warning(f"Low confidence ({conf:.4f}) in drift calculation at step {step + 1}")
        except Exception as e:
            logger.error(f"Could not calculate drift for step {step + 1}: {e}")
            # Decide how to handle errors
            # For now, we'll just record zeros.
            dx, dy, conf = Distance.from_nanometers(0), Distance.from_nanometers(0), 0.0

        # --- RECORD DATA ---
        # Calculate absolute actual position: base_position + measured_drift
        actual_x = start_x - dx
        actual_y = start_y - dy

        history.append({
            "timestamp": time.time(),
            "experiment_name": experiment_name,
            "iteration": iteration,
            "step": step + 1,
            "x_target_abs": target_x.nanometers,
            "y_target_abs": target_y.nanometers,
            "x_actual_abs": actual_x.nanometers,
            "y_actual_abs": actual_y.nanometers,
            "confidence": f"{conf:.4f}",
            "img_path": img_path.as_posix(),
        })

    # 5. Return to base position after trajectory is complete
    set_stage_xyr = cmd_factory.set_stage_xyr(start_x, start_y, R_ZERO)
    manager.send_command_blocking(set_stage_xyr, timeout=cfg.misc.command_timeout_ms)
    logger.info(f"Trajectory '{experiment_name}' finished. Returning to base position.")
    time.sleep(SETTLE_TIME)

    last_img_path, last_meta_path = scan(
        cfg, cmd_factory, manager, SEMScanMode.Slow1,
        save_suffix=f"{iteration:05d}_{experiment_name}_fin"
    )

    try:
        metadata = load_metadata(last_meta_path)
        (dx, dy), conf = calculate_drift(cfg, last_img_path, ref_img_path, metadata)
        dx = -dx
        logger.info(f"Drift detected: (dx={dx.nanometers}, dy={dy.nanometers}) with conf={conf:.4f}")
        if conf < 0.2:
            logger.warning(f"Low confidence ({conf:.4f}) in drift calculation at step final")
    except Exception as e:
        logger.error(f"Could not calculate drift for step final: {e}")
        # Decide how to handle errors
        # For now, we'll just record zeros.
        dx, dy, conf = Distance.from_nanometers(0), Distance.from_nanometers(0), 0.0

    actual_x = start_x - dx
    actual_y = start_y - dy
    history.append(
        {
            "timestamp": time.time(),
            "experiment_name": experiment_name,
            "iteration": iteration,
            "step": "end",
            "x_target_abs": start_x.nanometers,
            "y_target_abs": start_y.nanometers,
            "x_actual_abs": actual_x.nanometers,
            "y_actual_abs": actual_y.nanometers,
            "confidence": f"{conf:.4f}",
            "img_path": last_img_path.as_posix(),
        }
    )

    return history


# ==============================================================================
# SECTION 3: MAIN EXPERIMENT ORCHESTRATION
# ==============================================================================

def run(cfg: Settings, sem_cfg: FlexSEMSettings):
    logger.info("Starting hysteresis data collection script")
    sanity(cfg)

    # --- Setup SEM connection ---
    cmd_factory = MessageFactory(server_uid=sem_cfg.server.unit_id, client_uid=sem_cfg.client.unit_id)
    manager = CommunicationManager(sem_cfg)
    manager.connect()
    logger.info(f"Connected to {sem_cfg.server.unit_id} SEM device")

    current_pos = get_current_position(cfg, manager, cmd_factory)
    if not current_pos:
        logger.critical("Could not fetch initial stage position, aborting.")
        return

    base_x = current_pos['x']
    base_y = current_pos['y']

    # --- Define Experiment Parameters ---
    # These parameters define the grid of starting points for our experiments.
    grid_centers_x = [base_x]
    grid_centers_y = [base_y]

    # Phase 3: Generalization
    RANDOM_WALK_STEPS = 500

    MAGNIFICATIONS = [9000, 11000, 12000]
    MAG_MAX_STEPS = {
        9000: Distance.from_micrometers(6),
        11000: Distance.from_micrometers(4),
        12000: Distance.from_micrometers(3),
    }

    # --- Prepare for data logging ---
    history_filename = Path(time.strftime("hysteresis_dataset_%Y%m%d_%H%M%S.jsonl"))
    logger.info(f"Dataset will be saved to {history_filename}")

    all_experiments_data = []
    iteration_counter = 0

    try:
        # --- Main Experiment Loop ---
        for mag in MAGNIFICATIONS:
            for center_x in grid_centers_x:
                for center_y in grid_centers_y:
                    iteration_counter += 1
                    base_position = {'x': center_x, 'y': center_y}
                    logger.info(
                        f"--- Starting Iteration {iteration_counter} at Base: ({center_x.nanometers}, {center_y.nanometers}) ---")

                    max_step = MAG_MAX_STEPS[mag]

                    # --- PHASE 3: Random Walk ---
                    traj = generate_random_walk_2d(base_position, RANDOM_WALK_STEPS, max_step)
                    results = execute_and_record_trajectory(cfg, cmd_factory, manager, mag, traj, base_position,
                                                            "random_walk_2d", iteration_counter)
                    all_experiments_data.extend(results)

                    # Save progress after each major iteration
                    with open(history_filename, 'w') as f:
                        json.dump(all_experiments_data, f, indent=2)
                    logger.info(f"Progress for iteration {iteration_counter} saved to {history_filename}")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
    finally:
        # --- Final save and cleanup ---
        logger.info(f"Data collection finished. Final dataset saved to {history_filename}")
        with open(history_filename, 'w') as f:
            json.dump(all_experiments_data, f, indent=2)

        logger.info("All tasks executed, exiting.")
        manager.disconnect()


# ==============================================================================
# LEGACY FUNCTIONS (Kept for reference, but no longer used in `run`)
# ==============================================================================

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

