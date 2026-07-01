"""Per-dataset statistical profiling engine."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.logging_config import get_logger
from ml.intelligence.metadata_cache import ColumnMetadataCache
from ml.intelligence.models import ColumnProfile, DatasetProfile, ScannedDataset

logger = get_logger(__name__)


class StatisticsEngine:
    """Computes dataset- and column-level statistics."""

    def __init__(
        self,
        cache: ColumnMetadataCache | None = None,
        config: IntelligenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Initialize the statistics engine.

        Args:
            cache: Shared column metadata cache.
            config: Intelligence configuration.
        """
        self._cache = cache or ColumnMetadataCache(config=config)
        self._config = config

    def profile_all(
        self,
        frames: dict[str, pd.DataFrame],
        scanned: list[ScannedDataset],
    ) -> list[DatasetProfile]:
        """Build profiles for every discovered dataset.

        Args:
            frames: Loaded dataset frames.
            scanned: Scan metadata for each dataset.

        Returns:
            list[DatasetProfile]: Dataset profiles.
        """
        logger.info("statistics_engine_start", datasets=len(frames))
        start = time.perf_counter()
        scan_by_id = {item.dataset_id: item for item in scanned}
        profiles = [
            self.profile_dataset(dataset_id, frame, scan_by_id[dataset_id])
            for dataset_id, frame in frames.items()
            if dataset_id in scan_by_id
        ]
        logger.info(
            "statistics_engine_end",
            datasets=len(profiles),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return profiles

    def profile_dataset(
        self,
        dataset_id: str,
        frame: pd.DataFrame,
        scan: ScannedDataset,
    ) -> DatasetProfile:
        """Profile a single dataset.

        Args:
            dataset_id: Dataset identifier.
            frame: Loaded DataFrame, possibly sampled.
            scan: Scan metadata.

        Returns:
            DatasetProfile: Complete dataset profile.
        """
        row_count = scan.row_count
        duplicate_rows = int(frame.duplicated().sum())
        columns = [
            self._profile_column(dataset_id, frame, column, row_count)
            for column in frame.columns
        ]

        return DatasetProfile(
            dataset_id=dataset_id,
            filename=scan.filename,
            relative_path=scan.relative_path,
            row_count=row_count,
            column_count=len(frame.columns),
            memory_bytes=scan.memory_bytes,
            duplicate_row_count=duplicate_rows,
            columns=columns,
            column_groups=self._group_columns(columns),
        )

    def _profile_column(
        self,
        dataset_id: str,
        frame: pd.DataFrame,
        column: str,
        row_count: int,
    ) -> ColumnProfile:
        """Profile an individual column.

        Args:
            dataset_id: Dataset identifier.
            frame: Parent DataFrame.
            column: Column name.
            row_count: Total rows in the full dataset.

        Returns:
            ColumnProfile: Column statistics.
        """
        series = frame[column]
        col_name = str(column)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / len(series)) * 100, 2) if len(series) else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = (
            round((unique_count / len(series)) * 100, 2) if len(series) else 0.0
        )

        logical_dtype = self._cache.get_logical_dtype(dataset_id, col_name, series)
        is_datetime = self._cache.get_is_datetime(dataset_id, col_name, series)
        is_numeric = logical_dtype in {"integer", "float"}
        is_boolean = logical_dtype == "boolean" or self._is_boolean_like(series)
        is_text = logical_dtype == "string" and self._is_text_like(series)
        is_categorical = (
            logical_dtype == "string"
            and not is_text
            and unique_count <= max(50, len(series) * 0.2)
        )
        is_high_cardinality = unique_pct >= self._config.high_cardinality_ratio * 100

        stats: dict[str, Any] = {}
        if is_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            stats = {
                "min": self._safe_float(numeric.min()),
                "max": self._safe_float(numeric.max()),
                "mean": self._safe_float(numeric.mean()),
                "std": self._safe_float(numeric.std()),
            }

        sample_values = (
            series.dropna().astype(str).head(5).tolist()
            if null_count < len(series)
            else []
        )

        return ColumnProfile(
            name=col_name,
            inferred_dtype=logical_dtype,
            pandas_dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            duplicate_rows=int(series.duplicated().sum()),
            sample_values=sample_values,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_boolean=is_boolean,
            is_datetime=is_datetime,
            is_text=is_text,
            is_high_cardinality=is_high_cardinality,
            statistics=stats,
        )

    @staticmethod
    def _group_columns(columns: list[ColumnProfile]) -> dict[str, list[str]]:
        """Group column names by coarse type bucket."""
        groups: dict[str, list[str]] = {
            "numeric": [],
            "categorical": [],
            "boolean": [],
            "datetime": [],
            "text": [],
            "high_cardinality": [],
        }
        for col in columns:
            if col.is_numeric:
                groups["numeric"].append(col.name)
            if col.is_categorical:
                groups["categorical"].append(col.name)
            if col.is_boolean:
                groups["boolean"].append(col.name)
            if col.is_datetime:
                groups["datetime"].append(col.name)
            if col.is_text:
                groups["text"].append(col.name)
            if col.is_high_cardinality:
                groups["high_cardinality"].append(col.name)
        return {key: value for key, value in groups.items() if value}

    @staticmethod
    def _is_boolean_like(series: pd.Series) -> bool:
        """Detect binary columns with boolean-like values only."""
        if pd.api.types.is_bool_dtype(series):
            return True
        values = series.dropna().unique()
        if not (0 < len(values) <= 2):
            return False
        allowed = {
            True,
            False,
            0,
            1,
            0.0,
            1.0,
            "0",
            "1",
            "true",
            "false",
            "True",
            "False",
            "yes",
            "no",
        }
        return all(
            value in allowed or str(value).lower() in {str(a).lower() for a in allowed}
            for value in values
        )

    def _is_text_like(self, series: pd.Series) -> bool:
        """Detect free-text columns by average string length."""
        sample = series.dropna().astype(str).head(100)
        if sample.empty:
            return False
        return float(sample.str.len().mean()) >= self._config.text_avg_length

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert a numeric value to float when finite."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return round(float(value), 4)
