"""Utility helpers for the feature engineering module."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def validate_paths(*paths: Path | str) -> None:
    """Validate that all provided paths exist."""
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path_obj}")


def print_banner(title: str = "ETAIQ Feature Engineering Module") -> None:
    """Print a standard banner for module entry points."""
    print("=" * 40)
    print(title)
    print("=" * 40)
