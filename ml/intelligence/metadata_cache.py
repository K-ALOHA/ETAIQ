"""Cached column metadata to avoid repeated expensive computations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.datetime_detector import infer_is_datetime


@dataclass
class ColumnMetadataCache:
    """Caches per-column dtype and datetime inference results."""

    config: IntelligenceConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    _logical_dtypes: dict[tuple[str, str], str] = field(default_factory=dict)
    _is_datetime: dict[tuple[str, str], bool] = field(default_factory=dict)

    def get_logical_dtype(
        self,
        dataset_id: str,
        column_name: str,
        series: pd.Series,
    ) -> str:
        """Return cached logical dtype for a column.

        Args:
            dataset_id: Dataset identifier.
            column_name: Column name.
            series: Column data.

        Returns:
            str: Logical dtype label.
        """
        key = (dataset_id, column_name)
        if key in self._logical_dtypes:
            return self._logical_dtypes[key]

        dtype = self._infer_logical_dtype(column_name, series)
        self._logical_dtypes[key] = dtype
        return dtype

    def get_is_datetime(
        self,
        dataset_id: str,
        column_name: str,
        series: pd.Series,
    ) -> bool:
        """Return cached datetime flag for a column.

        Args:
            dataset_id: Dataset identifier.
            column_name: Column name.
            series: Column data.

        Returns:
            bool: Whether the column is datetime-like.
        """
        key = (dataset_id, column_name)
        if key in self._is_datetime:
            return self._is_datetime[key]

        is_dt = infer_is_datetime(column_name, series, self.config)
        self._is_datetime[key] = is_dt
        if is_dt:
            self._logical_dtypes[key] = "datetime"
        return is_dt

    def _infer_logical_dtype(self, column_name: str, series: pd.Series) -> str:
        """Infer logical dtype using cache-aware datetime detection.

        Args:
            column_name: Column name.
            series: Column data.

        Returns:
            str: Logical dtype name.
        """
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            return "float"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if infer_is_datetime(column_name, series, self.config):
            return "datetime"
        return "string"
