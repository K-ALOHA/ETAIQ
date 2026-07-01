"""Configuration parameters for the Cleaning Execution Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "ml" / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "ml" / "data" / "processed"
DEFAULT_REPORTS_DIR = REPO_ROOT / "ml" / "reports"


@dataclass(frozen=True)
class CleaningConfig:
    """Paths and settings for the Cleaning Engine."""

    raw_dir: Path = field(default=DEFAULT_RAW_DIR)
    processed_dir: Path = field(default=DEFAULT_PROCESSED_DIR)
    reports_dir: Path = field(default=DEFAULT_REPORTS_DIR)

    # Rollback configuration
    rollback_manifest_filename: str = "rollback_manifest.json"


DEFAULT_CLEANING_CONFIG = CleaningConfig()
