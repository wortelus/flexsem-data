# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import math
from pathlib import Path
from typing import Optional

from flexsem.commands.enum import SEMScanMode
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from corrector.config.metric import XYR, Distance


class BaseConfig(BaseModel):
    class Config:
        env_prefix = "FS1000_CORR_"
        env_nested_delimiter = "__"


class ExposureStepConfig(BaseModel):
    site_id: str = Field(min_length=1)
    # Possible to override for specific step
    exposure_mag: Optional[int] = None
    # Exposure time in seconds
    exposure_time_s: float = Field(gt=0)
    xyr: XYR

    @field_validator("site_id")
    def validate_site_id(cls, v):
        if not v.strip():
            raise ValueError("site_id must not be blank")
        return v


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

    @field_validator("steps")
    def validate_steps(cls, v):
        if not v:
            raise ValueError("at least one exposure step is required")
        return v


class FocusConfig(BaseModel):
    mag: int
    coarse: int
    fine: int


class PrecisionConfig(BaseModel):
    enabled: bool = False
    distance_threshold: Distance = Field(
        default_factory=lambda: Distance(value="100nm")
    )
    max_attempts: int = 3

    @field_validator("distance_threshold", mode="before")
    def parse_distance_threshold(cls, v):
        if isinstance(v, str):
            return Distance(value=Distance.convert_distance(v))
        return v

    @field_validator("distance_threshold")
    def validate_distance_threshold(cls, v):
        if v.value.nanometers < 0:
            raise ValueError("distance_threshold must be non-negative")
        return v

    @field_validator("max_attempts")
    def validate_max_attempts(cls, v):
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v


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
    precise: PrecisionConfig = Field(default_factory=PrecisionConfig)

    @field_validator("interval_s")
    def validate_interval(cls, v):
        if v <= 0:
            raise ValueError("interval_s must be positive")
        return v

    @field_validator("min_confidence")
    def validate_min_confidence(cls, v):
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError("min_confidence must be finite and between 0 and 1")
        return v

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


class UpdateReferenceConfig(BaseModel):
    precise: PrecisionConfig = Field(default_factory=PrecisionConfig)


class HysteresisConfig(BaseModel):
    enabled: bool
    force_warmup: bool = False
    simulator: bool
    model_path: Path
    sanity_limit: Distance
    warmup_distance: Distance

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
    command_beam_timeout_s: int
    command_timeout_s: int
    max_retries: int
    retry_delay_s: float
    reconnect_delay_s: float
    max_reconnect_retries: int
    turn_beam_off_on_error: bool


class Settings(BaseModel):
    drift: DriftCorrectionConfig
    exposure: ExposureConfig
    update_reference: UpdateReferenceConfig = Field(
        default_factory=UpdateReferenceConfig
    )
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
