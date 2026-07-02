"""Structured console logging for the training module."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any


class TrainingLogger:
    """Minimal structured logger for training components."""

    def __init__(self, name: str = "training", level: str = "INFO") -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def _log(self, level: int, message: str, **context: Any) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
            "logger": self.name,
            "message": message,
        }
        if context:
            payload["context"] = context
        self._logger.log(level, json.dumps(payload, default=str))

    def info(self, message: str, **context: Any) -> None:
        """Log an informational message."""
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, **context)

    def debug(self, message: str, **context: Any) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, message, **context)
