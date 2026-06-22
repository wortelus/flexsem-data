# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging

logger = logging.getLogger(__name__)

from pathlib import Path

import yaml

from corrector.config.models import Settings


def load_config(config_path: Path) -> Settings:
    """
    Loads configuration from a given YAML file path.

    Args:
        config_path: The explicit path to the settings.yaml file.

    Returns:
        A validated Settings object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    # Load main settings YAML
    if not config_path.exists():
        raise FileNotFoundError(f"Main configuration file not found: {config_path}")

    logger.debug(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        raw_settings = yaml.safe_load(f) or {}

    config = Settings(**raw_settings)
    logger.info(f"Loaded configuration from {config_path}")

    return config
