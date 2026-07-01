"""Schema version tracking and drift detection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ml.intelligence.logging_config import get_logger

logger = get_logger(__name__)


class VersionTracker:
    """Compares the current schema registry against a previous version."""

    def compare(
        self,
        current_registry: dict[str, Any],
        reports_dir: Path,
    ) -> dict[str, Any]:
        """Detect schema drift relative to a prior registry snapshot.

        Args:
            current_registry: Newly generated schema registry.
            reports_dir: Directory containing prior report artifacts.

        Returns:
            dict[str, Any]: Version comparison report.
        """
        logger.info("version_tracking_start")
        start = time.perf_counter()
        previous_path = reports_dir / "schema_registry.json"
        previous_registry = self._load_previous(previous_path)

        if previous_registry is None:
            report = {
                "status": "initial_version",
                "message": "No previous schema registry found; baseline created.",
                "changes": {},
            }
        else:
            report = {
                "status": "compared",
                "previous_generated_at": previous_registry.get("generated_at"),
                "current_generated_at": current_registry.get("generated_at"),
                "changes": self._diff_registries(previous_registry, current_registry),
            }

        logger.info(
            "version_tracking_end",
            status=report["status"],
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return report

    @staticmethod
    def _load_previous(path: Path) -> dict[str, Any] | None:
        """Load a previous schema registry if it exists.

        Args:
            path: Path to the prior registry file.

        Returns:
            dict[str, Any] | None: Previous registry or None.
        """
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _diff_registries(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Diff two schema registry documents.

        Args:
            previous: Prior registry.
            current: Current registry.

        Returns:
            dict[str, Any]: Structured diff report.
        """
        prev_datasets = previous.get("datasets", {})
        curr_datasets = current.get("datasets", {})

        added_datasets = sorted(set(curr_datasets) - set(prev_datasets))
        removed_datasets = sorted(set(prev_datasets) - set(curr_datasets))
        dataset_changes: dict[str, Any] = {}

        for dataset_id in sorted(set(prev_datasets) & set(curr_datasets)):
            changes = self._diff_dataset(
                prev_datasets[dataset_id], curr_datasets[dataset_id]
            )
            if changes:
                dataset_changes[dataset_id] = changes

        return {
            "added_datasets": added_datasets,
            "removed_datasets": removed_datasets,
            "dataset_changes": dataset_changes,
        }

    @staticmethod
    def _diff_dataset(
        previous: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        """Diff schema metadata for a single dataset.

        Args:
            previous: Prior dataset schema.
            current: Current dataset schema.

        Returns:
            dict[str, Any]: Dataset-level changes.
        """
        prev_cols = previous.get("columns", {})
        curr_cols = current.get("columns", {})
        added_columns = sorted(set(curr_cols) - set(prev_cols))
        removed_columns = sorted(set(prev_cols) - set(curr_cols))
        datatype_changes: dict[str, dict[str, str]] = {}

        for column in sorted(set(prev_cols) & set(curr_cols)):
            prev_dtype = prev_cols[column].get("logical_dtype")
            curr_dtype = curr_cols[column].get("logical_dtype")
            if prev_dtype != curr_dtype:
                datatype_changes[column] = {
                    "previous": prev_dtype,
                    "current": curr_dtype,
                }

        changes = {
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "datatype_changes": datatype_changes,
            "row_count_change": {
                "previous": previous.get("row_count"),
                "current": current.get("row_count"),
            },
        }
        return {key: value for key, value in changes.items() if value}
