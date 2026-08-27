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

from corrector.config.models import ExposureStepConfig, Settings
from corrector.config.sanity import sanity
from corrector.core.coordinates import StageOffset
from corrector.core.measurement import ReferenceFrame
from corrector.core.positioning import (
    PositionController,
    PositioningError,
    PrecisionThresholdError,
    RunState,
)
from corrector.core.scan import correction_scan
from corrector.core.step import stage_position_from_step
from corrector.core.task import (
    CorrectionTask,
    ExposureTask,
    PreciseCorrectionTask,
    PreciseUpdateReferenceTask,
    UpdateReferenceTask,
)
from corrector.hysteresis.manager import HysteresisManager
from corrector.hysteresis.warmup import run_warmup, warmup_linear
from corrector.utils.sem import (
    log_reported_motor_position,
    send_command_reliable_cfg,
)

logger = logging.getLogger(__name__)


def bootstrap_reference(
    cfg: Settings,
    manager: CommunicationManager,
    cmd_factory: MessageFactory,
    hysteresis: HysteresisManager,
) -> ReferenceFrame:
    first_step = cfg.exposure.steps[0]
    nominal_position = stage_position_from_step(first_step)

    if hysteresis.should_warmup():
        warmup_points = warmup_linear(
            hysteresis.sequence_length,
            (nominal_position.x, nominal_position.y),
            cfg.hysteresis.warmup_distance.value,
        )
        logger.info(
            "Generated %d warmup points ending at first site %s",
            len(warmup_points),
            first_step.site_id,
        )
        return run_warmup(
            cfg,
            manager,
            cmd_factory,
            hysteresis,
            warmup_points,
            first_step.site_id,
        )

    # With no image anchor yet, the first commanded nominal position defines
    # the origin of the local DIC frame.  Raw motor readback is diagnostic only.
    try:
        motor_command = hysteresis.compensate(nominal_position)
        set_stage_xyr = cmd_factory.set_stage_xyr(
            x=motor_command.x,
            y=motor_command.y,
            r=first_step.xyr.r.val,
        )
        send_command_reliable_cfg(
            set_stage_xyr,
            cfg=cfg,
            cmd_factory=cmd_factory,
            manager=manager,
        )
    except PositioningError:
        raise
    except Exception as exc:
        raise PositioningError(f"Initial stage movement failed: {exc}") from exc

    log_reported_motor_position(
        cfg,
        manager,
        cmd_factory,
        context="Initial",
    )
    try:
        image_path, metadata_path = correction_scan(
            cfg,
            cmd_factory,
            manager,
        )
    except Exception as exc:
        raise PositioningError(f"Initial reference scan failed: {exc}") from exc
    reference = ReferenceFrame(
        site_id=first_step.site_id,
        position=nominal_position,
        image_path=image_path,
        metadata_path=metadata_path,
    )
    logger.info(
        "Initial local DIC reference created for site %s at nominal x=%s y=%s",
        first_step.site_id,
        nominal_position.x,
        nominal_position.y,
    )
    return reference


def run_exposure_step(
    cfg: Settings,
    cmd_factory: MessageFactory,
    manager: CommunicationManager,
    state: RunState,
    controller: PositionController,
    step: ExposureStepConfig,
) -> None:
    """Position, commit a site reference, and expose all chunks for one step."""

    nominal_position = stage_position_from_step(step)
    rotation = step.xyr.r.val
    source_reference = state.reference_for_update(step.site_id)

    update_task_class = (
        PreciseUpdateReferenceTask
        if cfg.update_reference.precise.enabled
        else UpdateReferenceTask
    )
    update_task = update_task_class(cfg, cmd_factory, manager, step)
    try:
        final_observation = update_task.run(
            source_reference,
            nominal_position,
            rotation,
            controller,
        )
    except PrecisionThresholdError as exc:
        final_observation = exc.last_observation
        logger.warning(
            "PositioningError during precise reference update for site %s: %s; "
            "continuing with the last valid observation; final residual "
            "dx=%s dy=%s",
            step.site_id,
            exc,
            exc.position_error.dx,
            exc.position_error.dy,
        )

    # Commit only after the complete update task succeeds.  setdefault keeps a
    # site's first anchor stable while active_reference follows working scans.
    committed_reference = state.commit_reference(
        step.site_id,
        final_observation,
    )
    logger.info(
        "Committed active reference for site %s at x=%s y=%s",
        step.site_id,
        committed_reference.position.x,
        committed_reference.position.y,
    )

    # A successful reference update starts a fresh correction interval for
    # this exposure step.
    time_since_last_correction = 0.0
    remaining_exposure = float(step.exposure_time_s)
    epsilon = 1e-9

    while remaining_exposure > epsilon:
        time_to_next_correction = cfg.drift.interval_s - time_since_last_correction
        exposure_chunk = min(remaining_exposure, time_to_next_correction)
        chunk_step = step.model_copy(update={"exposure_time_s": exposure_chunk})
        ExposureTask(cfg, cmd_factory, manager, chunk_step).run()

        remaining_exposure -= exposure_chunk
        time_since_last_correction += exposure_chunk
        if remaining_exposure <= epsilon:
            break

        if time_since_last_correction + epsilon >= cfg.drift.interval_s:
            correction_task_class = (
                PreciseCorrectionTask if cfg.drift.precise.enabled else CorrectionTask
            )
            correction_task = correction_task_class(
                cfg,
                cmd_factory,
                manager,
                step,
            )
            try:
                correction_task.run(
                    state.active_reference,
                    nominal_position,
                    rotation,
                    controller,
                )
            except PrecisionThresholdError as exc:
                logger.warning(
                    "PositioningError during precise correction for site %s: %s; "
                    "continuing with the last valid observation; final residual "
                    "dx=%s dy=%s",
                    step.site_id,
                    exc,
                    exc.position_error.dx,
                    exc.position_error.dy,
                )
            time_since_last_correction = 0.0


def run(cfg: Settings, sem_cfg: FlexSEMSettings):
    logger.info("Starting flexsem-corrector")
    for index, step in enumerate(cfg.exposure.steps, start=1):
        logger.info(
            "Step %d site=%s: exposure time %ss at mag %s, correction mag %s",
            index,
            step.site_id,
            step.exposure_time_s,
            step.exposure_mag or cfg.exposure.exposure_mag,
            cfg.drift.correction_mag,
        )

    sanity(cfg)
    cmd_factory = MessageFactory(
        server_uid=sem_cfg.server.unit_id,
        client_uid=sem_cfg.client.unit_id,
    )
    manager = CommunicationManager(sem_cfg)
    manager.connect()
    logger.info("Connected to %s SEM device", sem_cfg.server.unit_id)

    hysteresis = HysteresisManager(cfg.hysteresis)
    try:
        first_reference = bootstrap_reference(
            cfg,
            manager,
            cmd_factory,
            hysteresis,
        )
        state = RunState(
            site_references={first_reference.site_id: first_reference},
            active_reference=first_reference,
            feedback_offset=StageOffset.zero(),
        )
        controller = PositionController(state, hysteresis)

        for index, step in enumerate(cfg.exposure.steps, start=1):
            logger.info(
                "Running exposure step %d/%d for site %s",
                index,
                len(cfg.exposure.steps),
                step.site_id,
            )
            run_exposure_step(
                cfg,
                cmd_factory,
                manager,
                state,
                controller,
                step,
            )
    except PositioningError as exc:
        # Earlier successful movements, feedback, and hysteresis history remain
        # valid.  The failing update is not committed and no subsequent
        # exposure is started.
        logger.error("Positioning failed; aborting before next exposure: %s", exc)
        raise
