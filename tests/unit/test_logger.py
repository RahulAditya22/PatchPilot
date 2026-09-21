from __future__ import annotations

import logging

from patchpilot.observability.logger import get_logger, setup_logging


def test_logger_setup_and_get() -> None:
    setup_logging(level="DEBUG")
    logger = get_logger("patchpilot.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "patchpilot.test"
    logger.info("Test log message")
