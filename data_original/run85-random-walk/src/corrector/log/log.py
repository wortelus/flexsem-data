# -*- coding: utf-8 -*-

# Copyright (c) 2025 Daniel Slavik @ EBEAM Centre. All rights reserved.
# This project is proprietary and confidential. Unauthorized copying
# of this file, via any medium is strictly prohibited.

__author__ = "Daniel Slavik"
__email__ = "daniel.slavik@wortelus.eu"

import logging
from logging.handlers import RotatingFileHandler

from corrector.config.models import LogConfig


def init_logger(log_cfg: LogConfig):
    """
    Sets up the application's logging configuration based on app_config.
    This function should be called once at application startup.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Get logging settings from config
    level = log_cfg.level
    path = log_cfg.path
    max_bytes = log_cfg.max_bytes
    backup_count = log_cfg.backup_count
    log_format = log_cfg.format

    # Ensure log directory exists
    log_dir = path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define a formatter
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Set level for console output
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (with rotation)
    file_handler = RotatingFileHandler(
        filename=path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging configured successfully.")
    root_logger.info(f"Log level: {level}, Log file: {path}")
