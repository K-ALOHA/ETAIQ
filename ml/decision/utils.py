"""Utility functions for Decision Intelligence Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.decision.logging_config import get_logger

logger = get_logger(__name__)


def load_report_json(reports_dir: Path, filename: str) -> dict[str, Any]:
    """Safely load a JSON report file, returning an empty dict if not found.

    Args:
        reports_dir: Directory containing reports.
        filename: Name of the JSON file to load.

    Returns:
        dict[str, Any]: Loaded JSON structure.
    """
    path = reports_dir / filename
    if not path.exists():
        logger.warning("report_file_missing", path=str(path))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        logger.error("report_file_corrupt", path=str(path), error=str(err))
        return {}


def dump_json(path: Path, payload: Any) -> None:
    """Safely write JSON to file with pretty printing."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("file_written", path=str(path))
    except Exception as err:
        logger.error("file_write_failed", path=str(path), error=str(err))
        raise
