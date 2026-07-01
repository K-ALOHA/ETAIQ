"""Datatype cleaner handler to parse and align column types."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class DatatypeCleaner:
    """Standardizes column data types (e.g. string to datetime, integer, float)."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        target_dtype: str,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Convert column type to target_dtype."""
        rows_before = len(df)
        if column not in df.columns:
            msg = f"Datatype cleaning failed: Column '{column}' not found in {dataset_id}."
            logger.error("datatype_clean_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        df_cleaned = df.copy()
        col_series = df_cleaned[column]
        dtype_str = target_dtype.lower()

        try:
            if "int" in dtype_str:
                # Convert to numeric, float, then Int64 (nullable integer type in pandas)
                numeric = pd.to_numeric(col_series, errors="coerce")
                df_cleaned[column] = numeric.round().astype("Int64")
            elif "float" in dtype_str or "double" in dtype_str:
                df_cleaned[column] = pd.to_numeric(col_series, errors="coerce")
            elif "date" in dtype_str or "time" in dtype_str:
                df_cleaned[column] = pd.to_datetime(col_series, errors="coerce")
            elif "bool" in dtype_str:
                # Map common strings to boolean
                mapping = {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "1.0": True,
                    "0.0": False,
                    "yes": True,
                    "no": False,
                }

                def to_bool(val: Any) -> Any:
                    if pd.isna(val):
                        return pd.NA
                    val_str = str(val).strip().lower()
                    return mapping.get(val_str, bool(val))

                df_cleaned[column] = col_series.map(to_bool).astype("boolean")
            else:
                df_cleaned[column] = col_series.astype(str)

            msg = f"Converted {dataset_id}.{column} to datatype {target_dtype}."
            logger.info("datatype_clean_completed", dataset_id=dataset_id, column=column, target_dtype=target_dtype)

            result = ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"target_dtype": target_dtype},
            )
            return df_cleaned, result

        except Exception as exc:
            msg = f"Datatype cleaning failed for {dataset_id}.{column} to {target_dtype}: {exc}"
            logger.exception("datatype_clean_failed", dataset_id=dataset_id, column=column, target_dtype=target_dtype)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
