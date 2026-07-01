"""Cleaning action executors mapping actions to their handlers.

This module routes deterministic cleaning actions to their respective handlers.
Only legitimate data CLEANING operations are supported here.

ML feature preparation (standardization, normalization, encoding) should NOT
happen in the cleaning engine - those operations belong in a separate
transformation module that runs after cleaning.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.datatype_cleaner import DatatypeCleaner
from ml.cleaning.duplicate_handler import DuplicateHandler
from ml.cleaning.gps_cleaner import GpsCleaner
from ml.cleaning.imputation import ImputationHandler
from ml.cleaning.integrity_cleaner import IntegrityCleaner
from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult
from ml.cleaning.outlier_handler import OutlierHandler
from ml.cleaning.timestamp_cleaner import TimestampCleaner

logger = get_logger(__name__)


class ActionExecutor:
    """Routes deterministic cleaning actions to the correct cleaning component.
    
    IMPORTANT: Only data CLEANING operations are supported:
    - Remove duplicates
    - Impute missing values
    - Fix data types
    - Clean timestamps
    - Validate GPS coordinates
    - Validate foreign keys
    - Remove outliers
    - Drop columns (when data quality dictates removal)
    
    NOT supported (belongs in transformation module):
    - Standardization/Normalization (feature scaling)
    - Encoding (categorical to numeric conversion)
    - Column renaming
    - Schema transformation
    """

    def __init__(self) -> None:
        self._imputation_handler = ImputationHandler()
        self._duplicate_handler = DuplicateHandler()
        self._datatype_cleaner = DatatypeCleaner()
        self._outlier_handler = OutlierHandler()
        self._gps_cleaner = GpsCleaner()
        self._timestamp_cleaner = TimestampCleaner()
        self._integrity_cleaner = IntegrityCleaner()

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        action_name: str,
        column: str | None,
        reference_keys: set[Any] | None = None,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Execute a cleaning action.

        Args:
            df: The input DataFrame.
            dataset_id: Identifier of the dataset.
            action_name: Verb like 'IMPUTE', 'DROP', 'ENCODE', etc.
            column: Target column or None.
            reference_keys: Reference primary keys for integrity check.
            details: Extra parameters from cleaning manifest.

        Returns:
            tuple[pd.DataFrame, ExecutorResult]: The cleaned DataFrame and result metadata.
        """
        rows_before = len(df)
        action = action_name.upper()

        if action in ("KEEP", "LEAVE_UNCHANGED"):
            msg = f"Recommendation is {action}; column {column} left unchanged."
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
            )

        if action == "REMOVE_DUPLICATES":
            return self._duplicate_handler.execute(df, dataset_id, column, **details)

        if action == "IMPUTE":
            if not column:
                msg = "Imputation action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            return self._imputation_handler.execute(df, dataset_id, column, **details)

        if action == "FIX_DATATYPE":
            if not column:
                msg = "FIX_DATATYPE action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            target_dtype = details.get("target_dtype", "string")
            return self._datatype_cleaner.execute(df, dataset_id, column, target_dtype, **details)

        if action == "STANDARDIZE_TIMESTAMP":
            if not column:
                msg = "STANDARDIZE_TIMESTAMP action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            return self._timestamp_cleaner.execute(df, dataset_id, column, **details)

        if action == "FIX_GPS":
            if not column:
                msg = "FIX_GPS action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            return self._gps_cleaner.execute(df, dataset_id, column, "FLAG", **details)

        if action == "REPAIR_FOREIGN_KEY":
            if not column:
                msg = "REPAIR_FOREIGN_KEY action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            return self._integrity_cleaner.execute(
                df,
                dataset_id,
                column,
                "DROP",
                reference_keys,
                **details,
            )

        if action == "TRIM_WHITESPACE":
            if not column:
                msg = "TRIM_WHITESPACE action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            df_cleaned = df.copy()
            df_cleaned[column] = df_cleaned[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            return df_cleaned, ExecutorResult(
                success=True,
                message=f"Trimmed whitespace in {dataset_id}.{column}.",
                records_before=rows_before,
                records_after=len(df_cleaned),
            )

        if action == "FORMAT_STRING":
            if not column:
                msg = "FORMAT_STRING action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            df_cleaned = df.copy()
            df_cleaned[column] = df_cleaned[column].apply(
                lambda value: str(value).strip() if not pd.isna(value) else value
            )
            return df_cleaned, ExecutorResult(
                success=True,
                message=f"Formatted string values in {dataset_id}.{column}.",
                records_before=rows_before,
                records_after=len(df_cleaned),
            )

        if action == "REMOVE_OUTLIERS":
            if not column:
                msg = "REMOVE_OUTLIERS action failed: column must be specified."
                return df, ExecutorResult(
                    success=False,
                    message=msg,
                    records_before=rows_before,
                    records_after=rows_before,
                    error_message=msg,
                )
            return self._outlier_handler.execute(df, dataset_id, column, **details)

        if action == "DROP":
            msg = (
                f"Skipped schema-changing DROP action for {dataset_id}.{column or 'dataset'}; "
                "cleaning preserves the original schema and only repairs values."
            )
            logger.info(
                "action_skipped",
                action=action,
                dataset_id=dataset_id,
                column=column,
                reason="schema_preservation",
            )
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"skipped_action": action},
            )

        if action == "FLAG":
            # Direct to coordinate, timestamp, or integrity handlers
            if column:
                # If GPS coordinate
                if any(w in column.lower() for w in ["lat", "lon", "latitude", "longitude"]):
                    return self._gps_cleaner.execute(df, dataset_id, column, action, **details)
                # If Foreign key key
                if any(w in column.lower() for w in ["id"]) and reference_keys is not None:
                    return self._integrity_cleaner.execute(df, dataset_id, column, action, reference_keys, **details)
                # If Timestamp
                if any(w in column.lower() for w in ["time", "date", "timestamp"]):
                    return self._timestamp_cleaner.execute(df, dataset_id, column, **details)

            msg = f"Flagged finding metadata for {dataset_id}.{column or 'dataset'}."
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
            )

        if action in {"STANDARDIZE", "NORMALIZE", "ENCODE", "ONE_HOT", "LABEL_ENCODING", "FEATURE_SCALING"}:
            msg = f"Skipped non-cleaning transformation action: {action}."
            logger.info("action_skipped", action=action, dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                details={"skipped_action": action},
            )

        msg = f"Unsupported action: {action}."
        logger.error("action_unsupported", action=action, dataset_id=dataset_id, column=column)
        return df, ExecutorResult(
            success=False,
            message=msg,
            records_before=rows_before,
            records_after=rows_before,
            error_message=msg,
        )
