# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from pathlib import Path
from typing import Optional

from flexsem.commands.enum import SEMScanMode
from pydantic import BaseModel, field_serializer, field_validator, model_validator

from corrector.config.metric import XYR


class BaseConfig(BaseModel):
    class Config:
        env_prefix = "FS1000_CORR_"
        env_nested_delimiter = "__"


class ExposureStepConfig(BaseModel):
    # Possible to override for specific step
    exposure_mag: Optional[int] = None
    correction_mag: Optional[int] = None
    # Exposure time in seconds
    exposure_time_s: int
    xyr: XYR


class ExposureConfig(BaseModel):
    exposure_type: SEMScanMode
    exposure_mag: int
    steps: list[ExposureStepConfig]

    @field_validator("exposure_type", mode="before")
    def convert_to_sem_scan_mode(cls, v):
        # Before validation
        if isinstance(v, str):
            # e.g. Fast1 -> SEMScanMode.Fast1
            return SEMScanMode[v]
        return v

    @field_serializer("exposure_type")
    def serialize_sem_scan_mode(self, scan: SEMScanMode):
        return scan.name


class FocusConfig(BaseModel):
    mag: int
    coarse: int
    fine: int


class DriftCorrectionConfig(BaseModel):
    correction_type: SEMScanMode
    correction_mag: int

    max_shift_nm_per_minute: float
    min_confidence: float
    interval_s: int

    # Ignoring the center exposure for drift correction
    ignore_center: bool
    ignore_center_radius_inner_fraction: float
    ignore_center_radius_outer_fraction: float

    @field_validator("correction_type", mode="before")
    def convert_to_sem_scan_mode(cls, v):
        # Before validation
        if isinstance(v, str):
            # e.g. Fast1 -> SEMScanMode.Fast1
            return SEMScanMode[v]
        return v

    @field_serializer("correction_type")
    def serialize_sem_scan_mode(self, scan: SEMScanMode):
        return scan.name


class LogConfig(BaseModel):
    level: str
    path: Path
    max_bytes: int
    backup_count: int
    format: str


class MiscConfig(BaseModel):
    sem_scan_path: Path
    sem_scan_image_filetype: str
    temp_path: Path
    command_timeout_ms: int
    beam_on_timeout_ms: int


class Settings(BaseModel):
    drift: DriftCorrectionConfig
    exposure: ExposureConfig
    focus: dict[int, FocusConfig]
    log: LogConfig
    misc: MiscConfig

    @field_validator("focus", mode="before")
    def transform_focus_list_to_dict(cls, v):
        if isinstance(v, list):
            return {item["mag"]: item for item in v}
        return v

    @model_validator(mode="after")
    def apply_default_magnifications(self):
        # Launched after validation of all fields
        for step in self.exposure.steps:
            if step.exposure_mag is None:
                step.exposure_mag = self.drift.exposure_mag
            if step.correction_mag is None:
                step.correction_mag = self.drift.correction_mag
        return self
