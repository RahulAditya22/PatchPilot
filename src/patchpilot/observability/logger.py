from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler


class StructuredFormatter(logging.Formatter):
    """Custom formatter to output logs as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            The JSON string representation of the log record.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Include any extra attributes added to the log record
        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_data[key] = value

        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Configure Python logging with structured JSON for file and Rich for console.

    Args:
        level: The logging level to use (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler using Rich
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(numeric_level)
    console_format = logging.Formatter("%(message)s", datefmt="[%X]")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler with JSON formatting and rotation (10MB max, 5 backups)
    file_handler = RotatingFileHandler("patchpilot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: The name of the logger.

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)
