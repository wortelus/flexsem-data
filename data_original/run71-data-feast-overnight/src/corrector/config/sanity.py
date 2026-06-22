# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

from corrector.config.models import Settings


def sanity(cfg: Settings):
    sanity_mkdir(cfg)


def sanity_mkdir(cfg: Settings):
    cfg.misc.temp_path.mkdir(parents=True, exist_ok=True)
