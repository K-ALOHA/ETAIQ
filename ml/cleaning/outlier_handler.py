"""Outlier handler to filter out anomalous and extreme records."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class OutlierHandler:
    """Detects and filters out outlier records based on standard deviation or range bounds."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Filter out rows containing numeric outliers in the specified column."""
        rows_before = len(df)
        # Exclude identifier columns from statistical outlier detection
        col_lower = column.lower() if column else ""
        is_identifier = any(
            col_lower == w or f"{w}_id" in col_lower or col_lower.endswith("_id")
            for w in ["id", "restaurant", "rider", "customer", "order"]
        )
        if is_identifier:
            msg = f"Skipped outlier removal on identifier column '{column}'."
            logger.info("outlier_clean_skipped_id_column", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"skipped": True, "reason": "identifier column"},
            )

        if column not in df.columns:
            msg = f"Outlier removal failed: Column '{column}' not found in {dataset_id}."
            logger.error("outlier_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        col_series = pd.to_numeric(df[column], errors="coerce")
        df_cleaned = df.copy()

        # Check for negative anomalies in positive-only columns
        is_positive_only = any(w in column.lower() for w in ["value", "capacity", "rating", "time", "min", "eta"])

        try:
            mean = col_series.mean()
            std = col_series.std()

            # If the column appears to be standardized (mean ≈ 0 and std ≈ 1), disable positive-only check
            # because negative z-scores are perfectly valid.
            if std and abs(mean) < 0.1 and abs(std - 1) < 0.1:
                is_positive_only = False

            filter_mask = pd.Series(True, index=df.index)

            if is_positive_only:
                # Flag negative numbers
                filter_mask = filter_mask & (col_series >= 0)

            if std and std > 0:
                # 3 standard deviations limit
                upper_limit = mean + 3 * std
                lower_limit = mean - 3 * std
                filter_mask = filter_mask & (col_series <= upper_limit) & (col_series >= lower_limit)
                strategy = f"3-std dev bounds [{round(lower_limit, 2)}, {round(upper_limit, 2)}]"
            else:
                strategy = "positive-only check (std is 0 or NaN)"

            # Apply filter
            df_cleaned = df_cleaned[filter_mask].copy()
            rows_after = len(df_cleaned)
            removed = rows_before - rows_after

            msg = f"Removed {removed} outlier rows from {dataset_id}.{column} using {strategy}."
            logger.info(
                "remove_outliers_completed",
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
                details={"removed_count": removed, "strategy": strategy},
            )
            return df_cleaned, result

        except Exception as exc:
            msg = f"Outlier removal failed for {dataset_id}.{column}: {exc}"
            logger.exception("remove_outliers_failed", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
