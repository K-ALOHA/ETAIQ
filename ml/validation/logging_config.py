"""Structured logging configuration for the ML validation engine."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_validation_logging(log_level: str = "INFO") -> None:
    """Configure structlog for validation pipeline output.

    Args:
        log_level: Logging verbosity (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    """
    global _configured
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """Return a structured logger for the given module name.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A structlog bound logger instance.
    """
    return structlog.get_logger(name)
