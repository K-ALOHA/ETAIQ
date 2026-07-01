"""Normalization handler to perform scaling and categorical encoding."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class NormalizationHandler:
    """Handles feature scaling (STANDARDIZE, NORMALIZE) and categorical encoding (ENCODE)."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        action: str,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Perform scaling or encoding on the specified column."""
        rows_before = len(df)
        if column not in df.columns:
            msg = f"{action} failed: Column '{column}' not found in {dataset_id}."
            logger.error("normalization_column_missing", action=action, dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        df_cleaned = df.copy()
        col_series = df_cleaned[column]

        try:
            if action == "STANDARDIZE":
                mean = col_series.mean()
                std = col_series.std()
                if std > 0:
                    df_cleaned[column] = (col_series - mean) / std
                    strategy = f"StandardScaler (mean={round(mean, 4)}, std={round(std, 4)})"
                else:
                    strategy = "StandardScaler (std is 0; left unchanged)"
            elif action == "NORMALIZE":
                col_min = col_series.min()
                col_max = col_series.max()
                diff = col_max - col_min
                if diff > 0:
                    df_cleaned[column] = (col_series - col_min) / diff
                    strategy = f"MinMaxScaler (min={round(col_min, 4)}, max={round(col_max, 4)})"
                else:
                    strategy = "MinMaxScaler (min == max; left unchanged)"
            elif action == "ENCODE":
                # Convert categorical text to label codes to maintain column structure
                codes = col_series.astype("category").cat.codes
                # cat.codes is -1 for missing values. Let's make sure it handles that.
                df_cleaned[column] = codes
                enc_type = details.get("encoding", "label_encoding")
                strategy = f"Label Encoding (type={enc_type})"
            else:
                msg = f"Unknown normalization action: {action}."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )

            msg = f"Applied {action} on {dataset_id}.{column} using {strategy}."
            logger.info("normalization_completed", action=action, dataset_id=dataset_id, column=column, strategy=strategy)

            result = ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"strategy": strategy, "action": action},
            )
            return df_cleaned, result

        except Exception as exc:
            msg = f"Normalization action {action} failed for {dataset_id}.{column}: {exc}"
            logger.exception("normalization_failed", action=action, dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
