"""Structured logging.

A library must not configure the root logger on import. :func:`configure_logging`
exists for applications and the CLI to opt in; everything else just calls
:func:`get_logger`.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Render records as one JSON object per line, including extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON representation of ``record``."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int | str = logging.INFO, *, json_output: bool = False) -> None:
    """Configure the ``toolbroker`` logger. For applications and the CLI only."""
    logger = logging.getLogger("toolbroker")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JSONFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(f"toolbroker.{name}")
