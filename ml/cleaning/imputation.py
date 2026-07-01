"""Imputation handler to fill missing values in columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class ImputationHandler:
    """Imputes missing values using statistics or placeholders."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Fill missing values in the specified column.

        If the column is numeric, impute with median or mean.
        If it's categorical/string, impute with mode or a constant fallback.
        """
        rows_before = len(df)
        if column not in df.columns:
            msg = f"Imputation failed: Column '{column}' not found in {dataset_id}."
            logger.error("impute_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        col_series = df[column]
        null_count = int(col_series.isna().sum())

        if null_count == 0:
            msg = f"No missing values in {dataset_id}.{column}; left unchanged."
            logger.info("impute_no_nulls", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"null_count_filled": 0},
            )

        df_cleaned = df.copy()

        # Determine logical type of series
        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        is_bool = pd.api.types.is_bool_dtype(col_series)

        impute_value: Any = None
        strategy = ""

        if is_numeric and not is_bool:
            # Impute numeric with median
            impute_value = col_series.median()
            if pd.isna(impute_value):
                impute_value = 0.0
            strategy = f"median ({impute_value})"
        elif is_bool:
            # Impute boolean with most frequent value (mode)
            mode_series = col_series.mode()
            impute_value = bool(mode_series.iloc[0]) if not mode_series.empty else False
            strategy = f"mode ({impute_value})"
        else:
            # Impute string/categorical with mode
            mode_series = col_series.mode()
            impute_value = str(mode_series.iloc[0]) if not mode_series.empty else "Unknown"
            strategy = f"mode ({impute_value})"

        df_cleaned[column] = col_series.fillna(impute_value)

        msg = f"Imputed {null_count} nulls in {dataset_id}.{column} using {strategy}."
        logger.info(
            "impute_completed",
            dataset_id=dataset_id,
            column=column,
            null_count=null_count,
            strategy=strategy,
        )

        result = ExecutorResult(
            success=True,
            message=msg,
            records_before=rows_before,
            records_after=rows_before,
            details={"null_count_filled": null_count, "impute_value": impute_value, "strategy": strategy},
        )
        return df_cleaned, result
