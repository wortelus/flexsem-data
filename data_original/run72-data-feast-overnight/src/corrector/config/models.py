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

from corrector.config.metric import XYR, Distance


class BaseConfig(BaseModel):
    class Config:
        env_prefix = "FS1000_CORR_"
        env_nested_delimiter = "__"


class ExposureStepConfig(BaseModel):
    # Possible to override for specific step
    exposure_mag: Optional[int] = None
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

    # Ignoring the center/borders exposure for drift correction
    ignore_center_x_pix: int
    ignore_center_y_pix: int
    ignore_bottom_pix: int

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


class HysteresisConfig(BaseModel):
    enabled: bool
    simulator: bool
    model_path: Path
    sanity_limit: Distance
    warmup_distance: Distance
    warmup_point: XYR

    @field_validator("sanity_limit", "warmup_distance", mode="before")
    def parse_distance_rotation(cls, v, field):
        if isinstance(v, str):
            if field.field_name in ("sanity_limit", "warmup_distance"):
                return Distance(value=Distance.convert_distance(v))
        return v


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
    beam_on_timeout_s: int
    command_timeout_s: int
    max_retries: int
    retry_delay_s: float
    reconnect_delay_s: float
    max_reconnect_retries: int
    turn_beam_off_on_error: bool


class Settings(BaseModel):
    drift: DriftCorrectionConfig
    exposure: ExposureConfig
    hysteresis: HysteresisConfig
    log: LogConfig
    misc: MiscConfig

    @model_validator(mode="after")
    def apply_default_magnifications(self):
        # Launched after validation of all fields
        for step in self.exposure.steps:
            if step.exposure_mag is None:
                step.exposure_mag = self.exposure.exposure_mag
        return self
