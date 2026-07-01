"""GPS cleaner to filter out-of-bounds geographic coordinate rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class GpsCleaner:
    """Filters out geographic latitude/longitude coordinate violations."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        action: str = "DROP",
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Filter out or nullify records with invalid latitude or longitude coordinates."""
        rows_before = len(df)
        if column not in df.columns:
            msg = f"GPS cleaning failed: Column '{column}' not found in {dataset_id}."
            logger.error("gps_clean_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        col_series = pd.to_numeric(df[column], errors="coerce")
        df_cleaned = df.copy()

        try:
            # Detect coordinate type from column name heuristics
            is_lat = any(w in column.lower() for w in ["lat", "latitude"])
            is_lon = any(w in column.lower() for w in ["lon", "lng", "longitude"])

            if is_lat:
                # Valid latitude bounds: [-90.0, 90.0]
                filter_mask = col_series.isna() | ((col_series >= -90.0) & (col_series <= 90.0))
                bounds = "[-90.0, 90.0]"
            elif is_lon:
                # Valid longitude bounds: [-180.0, 180.0]
                filter_mask = col_series.isna() | ((col_series >= -180.0) & (col_series <= 180.0))
                bounds = "[-180.0, 180.0]"
            else:
                msg = f"GPS cleaning skipped: Column '{column}' does not appear to be latitude or longitude."
                logger.warning("gps_clean_not_gps", dataset_id=dataset_id, column=column)
                return df, ExecutorResult(
                    success=True,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                )

            invalid_mask = ~filter_mask
            invalid_count = int(invalid_mask.sum())

            if action.upper() == "FLAG":
                # NULLIFY: Set out-of-bounds values to NaN (NULL)
                df_cleaned.loc[invalid_mask, column] = pd.NA
                rows_after = len(df_cleaned)
                msg = f"Nullified {invalid_count} out-of-bounds GPS values in {dataset_id}.{column}."
                logger.info(
                    "gps_clean_completed",
                    dataset_id=dataset_id,
                    column=column,
                    rows_before=rows_before,
                    rows_after=rows_after,
                    removed=0,
                )
                result = ExecutorResult(
                    success=True,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_after,
                    details={"corrected_count": invalid_count, "bounds": bounds},
                )
            else:
                # REMOVE: Drop rows containing out-of-bounds values
                df_cleaned = df_cleaned[filter_mask].copy()
                rows_after = len(df_cleaned)
                removed = rows_before - rows_after
                msg = f"Removed {removed} rows with out-of-bound GPS values in {dataset_id}.{column} (bounds={bounds})."
                logger.info(
                    "gps_clean_completed",
                    dataset_id=dataset_id,
                    column=column,
                    rows_before=rows_before,
                    rows_after=rows_after,
                    removed=removed,
                )
                result = ExecutorResult(
                    success=True,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_after,
                    details={"removed_count": removed, "bounds": bounds},
                )
            return df_cleaned, result

        except Exception as exc:
            msg = f"GPS cleaning failed for {dataset_id}.{column}: {exc}"
            logger.exception("gps_clean_failed", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
