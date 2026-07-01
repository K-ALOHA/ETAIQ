"""Timestamp cleaner to standardize datetime formats."""

from __future__ import annotations

from datetime import datetime
import numbers
from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class TimestampCleaner:
    """Parses and standardizes messy datetime timestamp columns."""

    @staticmethod
    def _parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
        if value is None or pd.isna(value):
            return pd.NaT

        if isinstance(value, pd.Timestamp):
            return value

        if isinstance(value, datetime):
            return pd.Timestamp(value)

        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            numeric_value = float(value)
            if abs(numeric_value) >= 10**12:
                return pd.to_datetime(numeric_value, unit="ms", errors="coerce")
            return pd.to_datetime(numeric_value, unit="s", errors="coerce")

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return pd.NaT

            if text.lstrip("+-").isdigit():
                try:
                    numeric_value = int(text)
                    if abs(numeric_value) >= 10**12:
                        return pd.to_datetime(numeric_value, unit="ms", errors="coerce")
                    return pd.to_datetime(numeric_value, unit="s", errors="coerce")
                except (TypeError, ValueError):
                    return pd.NaT

            if "/" in text:
                for fmt in (
                    "%d/%m/%Y %H:%M",
                    "%d/%m/%Y %H:%M:%S",
                    "%d/%m/%Y",
                ):
                    try:
                        return pd.to_datetime(text, format=fmt, errors="coerce")
                    except (TypeError, ValueError):
                        continue

            try:
                return pd.to_datetime(text, errors="coerce")
            except (TypeError, ValueError):
                return pd.NaT

        return pd.NaT

    @staticmethod
    def _format_timestamp(value: pd.Timestamp | pd.NaT) -> str | None:
        if pd.isna(value):
            return None

        timestamp = pd.Timestamp(value)
        if timestamp.tz is not None:
            return timestamp.isoformat(timespec="seconds").replace("T", " ")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Convert a timestamp column to a standardized datetime string format."""
        rows_before = len(df)
        if column not in df.columns:
            msg = f"Timestamp cleaning failed: Column '{column}' not found in {dataset_id}."
            logger.error("timestamp_clean_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        df_cleaned = df.copy()

        try:
            parsed = df_cleaned[column].apply(self._parse_timestamp)
            df_cleaned[column] = parsed.apply(self._format_timestamp)

            null_count = int(df_cleaned[column].isna().sum())
            msg = f"Standardized timestamp column {dataset_id}.{column}. {null_count} values could not be parsed."
            logger.info("timestamp_clean_completed", dataset_id=dataset_id, column=column, unparsed_nulls=null_count)

            result = ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"unparsed_nulls": null_count},
            )
            return df_cleaned, result

        except Exception as exc:
            msg = f"Timestamp cleaning failed for {dataset_id}.{column}: {exc}"
            logger.exception("timestamp_clean_failed", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
