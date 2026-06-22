# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import math

from flexsem.utils.metric import Distance as SafeDistance
from flexsem.utils.metric import Rotation as SafeRotation
from pydantic import BaseModel, field_validator

from corrector.utils.utils import is_valid_int


def _strip(v: str, suffix: str) -> str:
    return v.replace(suffix, "").strip()


def _strip_float(v, suffix: str) -> float:
    return float(_strip(v, suffix))


def _strip_int(v, suffix: str) -> int:
    return int(_strip(v, suffix))


class Distance(BaseModel):
    value: SafeDistance

    @field_validator("value", mode="before")
    def convert_distance(cls, v):
        if isinstance(v, SafeDistance):
            return v
        if isinstance(v, str):
            if v.endswith("nm"):
                # allow only ints
                if not is_valid_int(_strip(v, "nm")):
                    raise ValueError("Nanometers must be an integer value")
                return SafeDistance.from_nanometers(
                    _strip_int(v, "nm"),
                )
            elif v.endswith("um"):
                return SafeDistance.from_micrometers(_strip_float(v, "um"))
            elif v.endswith("mm"):
                return SafeDistance.from_millimeters(_strip_float(v, "mm"))
        raise ValueError("Invalid distance format")


class Rotation(BaseModel):
    val: SafeRotation

    @field_validator("val", mode="before")
    def convert_rotation(cls, v):
        if isinstance(v, SafeRotation):
            return v
        if isinstance(v, str):
            if v.endswith("rad"):
                return SafeRotation.from_radians(_strip_float(v, "rad"))
            if v.endswith("deg"):
                return SafeRotation.from_degrees(_strip_float(v, "deg"))
        raise ValueError("Invalid rotation format")

    @property
    def radians(self) -> float:
        return math.radians(self.degrees)

    def to_safe_dataclass(self) -> SafeRotation:
        return SafeRotation.from_degrees(self.degrees)


class XYR(BaseModel):
    x: Distance
    y: Distance
    r: Rotation

    @classmethod
    def zeroes(cls) -> "XYR":
        return cls(
            x=Distance(value=SafeDistance.from_nanometers(0)),
            y=Distance(value=SafeDistance.from_nanometers(0)),
            r=Rotation(val=SafeRotation.from_degrees(0.0)),
        )

    @field_validator("x", "y", "r", mode="before")
    def parse_distance_rotation(cls, v, field):
        if isinstance(v, str):
            if field.field_name in ("x", "y"):
                return Distance(value=Distance.convert_distance(v))
            elif field.field_name == "r":
                return Rotation(val=Rotation.convert_rotation(v))
        return v
