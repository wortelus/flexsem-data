# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import copy
import enum
import logging
from pathlib import Path
from time import sleep
from typing import List

from flexsem.commands import MessageFactory
from flexsem.state.manager import CommunicationManager

from corrector.config.models import DriftCorrectionConfig, ExposureStepConfig, Settings
from corrector.core.dic import dic_measure
from corrector.core.scan import correction_scan, exposure_scan
from corrector.misc.metadata import load_metadata

logger = logging.getLogger(__name__)


class TaskType(enum.Enum):
    EXPOSURE = "exposure"
    CORRECTION = "correction"
    UPDATE_REFERENCE = "update_reference"


class BaseTask:
    task_type: TaskType
    step: ExposureStepConfig

    def __init__(
            self,
            cfg: Settings,
            cmd_factory: MessageFactory,
            manager: CommunicationManager,
    ):
        self.cfg = cfg
        self.cmd_factory = cmd_factory
        self.manager = manager

    def scan(self, ref_img_path: Path = None) -> ((float, float, float), Path, Path):
        """
        Abstract method to perform the scan associated with the task.
        """
        raise NotImplementedError

    def execute(self):
        """
        Abstract method to execute the task.
        """
        raise NotImplementedError


class ExposureTask(BaseTask):
    task_type: TaskType = TaskType.EXPOSURE
    step: ExposureStepConfig

    def __init__(
            self,
            cfg: Settings,
            cmd_factory: MessageFactory,
            manager: CommunicationManager,
            step: ExposureStepConfig,
    ):
        super().__init__(cfg, cmd_factory, manager)
        self.cfg = cfg
        self.cmd_factory = cmd_factory
        self.manager = manager
        self.step = step

    def scan(self, ref_img_path: Path = None) -> ((float, float, float), Path, Path):
        """
        Perform the scan associated with the exposure task.
        """
        # Magnify -> Focus -> Scan
        img_path, metadata_path = correction_scan(self.cfg, self.cmd_factory, self.manager, settle_time_s=3)

        # Calculate movement to reference image
        metadata = load_metadata(metadata_path)
        (dx, dy), conf = dic_measure(self.cfg, ref_img_path, img_path, metadata)
        logger.info(f"Exposure task movement at (x={dx}, y={dy}) with conf={conf} to the reference image")

        return (dx, dy, conf), img_path, metadata_path

    def execute(self):
        """
        Execute the exposure task.
        """
        logger.info(f"Executing exposure task for {self.step.exposure_time_s}s")

        # Magnify -> Focus -> Scan
        exposure_scan(self.cfg, self.cmd_factory, self.manager, self.step)

        # Exposure (simulated with sleep)
        # @TODO: something else ?
        sleep(self.step.exposure_time_s)


class CorrectionTask(BaseTask):
    task_type: TaskType = TaskType.CORRECTION
    step: ExposureStepConfig
    correction: DriftCorrectionConfig

    def __init__(
            self,
            cfg: Settings,
            cmd_factory: MessageFactory,
            manager: CommunicationManager,
            step: ExposureStepConfig,
            correction: DriftCorrectionConfig):
        super().__init__(cfg, cmd_factory, manager)
        self.step = step
        self.correction = correction

    def scan(self, ref_img_path: Path = None) -> ((float, float, float), Path, Path):
        """
        Perform the scan associated with the correction task.
        """
        logger.info(f"Performing correction scan")

        # Magnify -> Focus -> Scan
        img_path, metadata_path = correction_scan(self.cfg, self.cmd_factory, self.manager, settle_time_s=3)

        # Calculate drift
        metadata = load_metadata(metadata_path)
        (dx, dy), conf = dic_measure(self.cfg, ref_img_path, img_path, metadata)
        logger.info(f"Drift detected at (x={dx}, y={dy}) with conf={conf} to the reference image")

        return (dx, dy, conf), img_path, metadata_path

    def execute(self):
        pass


class UpdateReferenceTask(BaseTask):
    task_type: TaskType = TaskType.UPDATE_REFERENCE
    step: ExposureStepConfig
    correction: DriftCorrectionConfig

    def __init__(
            self,
            cfg: Settings,
            cmd_factory: MessageFactory,
            manager: CommunicationManager,
            step: ExposureStepConfig,
            correction: DriftCorrectionConfig,
    ):
        super().__init__(cfg, cmd_factory, manager)
        self.step = step
        self.correction = correction

    def scan(self, ref_img_path: Path = None) -> ((float, float, float), Path, Path):
        """
        Perform the scan associated with the update reference task.
        """
        logger.info(f"Performing reference update scan")

        # Magnify -> Focus -> Scan
        img_path, metadata_path = correction_scan(self.cfg, self.cmd_factory, self.manager, settle_time_s=5)

        metadata = load_metadata(metadata_path)
        (dx, dy), conf = dic_measure(self.cfg, ref_img_path, img_path, metadata, override_mask_disable=True)
        logger.info(
            f"Reference image update drift measured at (x={dx}, y={dy}) with conf={conf} to the original reference image")

        return (dx, dy, conf), img_path, metadata_path

    def execute(self):
        pass


def build_task_list(cfg: Settings,
                    cmd_factory: MessageFactory,
                    manager: CommunicationManager) -> List[BaseTask]:
    final_task_list: List[BaseTask] = []
    correction_interval = cfg.drift.interval_s

    time_since_last_correction = 0.0
    for step in cfg.exposure.steps:
        remaining_exposure = step.exposure_time_s

        # Initial reference update task
        final_task_list.append(UpdateReferenceTask(cfg, cmd_factory, manager, step=step, correction=cfg.drift))

        while remaining_exposure > 0:
            # Time until next correction
            time_to_next_correction = correction_interval - time_since_last_correction

            # Either finish the current exposure, or attempt a correction
            exposure_chunk = min(remaining_exposure, time_to_next_correction)

            # Add exposure task for the chunk
            if exposure_chunk > 1e-9:
                # @TODO: join small exposures ?
                chunk_step = copy.deepcopy(step)
                chunk_step.exposure_time_s = exposure_chunk
                final_task_list.append(ExposureTask(cfg, cmd_factory, manager, step=chunk_step))

            # Update remaining exposure and time since last correction
            remaining_exposure -= exposure_chunk
            time_since_last_correction += exposure_chunk

            # If we've reached the correction interval
            if time_since_last_correction >= correction_interval:
                # Append correction task
                final_task_list.append(CorrectionTask(cfg, cmd_factory, manager, step=step, correction=cfg.drift))
                # Reset the timer
                time_since_last_correction = 0.0

    return final_task_list
