"""Dynamic schema registry built from discovered datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ml.intelligence.logging_config import get_logger
from ml.intelligence.metadata_cache import ColumnMetadataCache
from ml.intelligence.models import ScannedDataset

logger = get_logger(__name__)


class SchemaRegistry:
    """Builds and maintains a dynamic schema registry from DataFrames."""

    def __init__(self, cache: ColumnMetadataCache | None = None) -> None:
        """Initialize the registry builder.

        Args:
            cache: Shared column metadata cache.
        """
        self._cache = cache or ColumnMetadataCache()

    def build(
        self,
        frames: dict[str, pd.DataFrame],
        scanned: list[ScannedDataset],
    ) -> dict[str, Any]:
        """Construct a schema registry payload for all datasets.

        Args:
            frames: Loaded dataset frames keyed by dataset id.
            scanned: Scan metadata for each dataset.

        Returns:
            dict[str, Any]: Schema registry document.
        """
        logger.info("schema_registry_build_start", datasets=len(frames))
        scan_by_id = {item.dataset_id: item for item in scanned}
        datasets: dict[str, Any] = {}

        for dataset_id, frame in frames.items():
            scan = scan_by_id.get(dataset_id)
            columns: dict[str, Any] = {}
            for column in frame.columns:
                series = frame[column]
                col_name = str(column)
                logical_dtype = self._cache.get_logical_dtype(
                    dataset_id, col_name, series
                )
                columns[col_name] = {
                    "logical_dtype": logical_dtype,
                    "pandas_dtype": str(series.dtype),
                    "nullable": bool(series.isna().any()),
                    "unique_count": int(series.nunique(dropna=True)),
                }

            datasets[dataset_id] = {
                "filename": scan.filename if scan else f"{dataset_id}.csv",
                "relative_path": scan.relative_path if scan else "",
                "row_count": scan.row_count if scan else len(frame),
                "column_count": len(frame.columns),
                "columns": columns,
            }

        registry = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_count": len(datasets),
            "datasets": datasets,
        }
        logger.info("schema_registry_build_end", datasets=len(datasets))
        return registry

    @staticmethod
    def column_names(registry: dict[str, Any], dataset_id: str) -> list[str]:
        """Return column names for a dataset from a registry document.

        Args:
            registry: Schema registry payload.
            dataset_id: Dataset identifier.

        Returns:
            list[str]: Column names.
        """
        dataset = registry.get("datasets", {}).get(dataset_id, {})
        return list(dataset.get("columns", {}).keys())
