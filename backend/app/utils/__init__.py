"""Shared utility functions."""

from app.core.logging import get_logger

logger = get_logger(__name__)


def to_snake_case(value: str) -> str:
    """Convert a string to snake_case format.

    Args:
        value: Input string to normalize.

    Returns:
        str: Snake-cased representation of the input.
    """
    return value.strip().lower().replace(" ", "_").replace("-", "_")
