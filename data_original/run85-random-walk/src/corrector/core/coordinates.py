# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

from __future__ import annotations

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import math
from dataclasses import dataclass

from flexsem.utils import Distance as SEMDistance


@dataclass(frozen=True)
class StagePosition:
    """DIC/reference-relative XY position in SEM stage coordinates."""

    x: SEMDistance
    y: SEMDistance

    @classmethod
    def from_nanometers(cls, x_nm: int, y_nm: int) -> "StagePosition":
        return cls(
            x=SEMDistance.from_nanometers(x_nm),
            y=SEMDistance.from_nanometers(y_nm),
        )

    def shifted(self, offset: "StageOffset") -> "StagePosition":
        return StagePosition(x=self.x + offset.dx, y=self.y + offset.dy)

    def minus(self, other: "StagePosition") -> "StageOffset":
        return StageOffset(dx=self.x - other.x, dy=self.y - other.y)


@dataclass(frozen=True)
class StageOffset:
    """XY displacement in SEM stage coordinates."""

    dx: SEMDistance
    dy: SEMDistance

    @classmethod
    def from_nanometers(cls, dx_nm: int, dy_nm: int) -> "StageOffset":
        return cls(
            dx=SEMDistance.from_nanometers(dx_nm),
            dy=SEMDistance.from_nanometers(dy_nm),
        )

    @classmethod
    def zero(cls) -> "StageOffset":
        return cls.from_nanometers(0, 0)

    def __add__(self, other: "StageOffset") -> "StageOffset":
        if not isinstance(other, StageOffset):
            return NotImplemented
        return StageOffset(dx=self.dx + other.dx, dy=self.dy + other.dy)

    def __neg__(self) -> "StageOffset":
        return StageOffset(dx=-self.dx, dy=-self.dy)

    def distance_nm(self) -> float:
        return math.hypot(self.dx.nanometers, self.dy.nanometers)
