"""Integrity cleaner to resolve and flag referential integrity/orphan keys."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class IntegrityCleaner:
    """Handles referential integrity violations by flagging or dropping orphan foreign key records."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str,
        action: str,
        reference_keys: set[Any] | None = None,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Flag or drop records where the foreign key column contains values not in reference_keys."""
        rows_before = len(df)
        if column not in df.columns:
            msg = f"Integrity cleaning failed: Column '{column}' not found in {dataset_id}."
            logger.error("integrity_clean_column_missing", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )

        if reference_keys is None:
            # If no primary keys provided, we cannot identify orphans; leave unchanged
            msg = f"Skipped integrity cleaning for {dataset_id}.{column}: No reference primary keys provided."
            logger.warning("integrity_clean_missing_ref_keys", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
            )

        df_cleaned = df.copy()
        col_series = df_cleaned[column]

        try:
            # Clean coordinate/identifier datatype to match keys (convert floats like '1.0' to '1')
            def normalize_key(val: Any) -> Any:
                if pd.isna(val):
                    return None
                try:
                    f = float(val)
                    if f.is_integer():
                        return str(int(f))
                    return str(f)
                except (ValueError, TypeError):
                    return str(val).strip()

            normalized_ref_keys = {normalize_key(k) for k in reference_keys if k is not None}
            col_normalized = col_series.map(normalize_key)

            # Identify orphans
            is_orphan = col_normalized.notna() & (~col_normalized.isin(normalized_ref_keys))
            orphan_count = int(is_orphan.sum())

            # Write the normalized keys back to the dataframe column so they are cleaned (e.g. 5764.0 -> 5764)
            df_cleaned[column] = col_normalized

            if action == "DROP":
                if orphan_count > 0:
                    df_cleaned = df_cleaned[~is_orphan].copy()
                rows_after = len(df_cleaned)
                msg = f"Dropped {orphan_count} orphan key rows from {dataset_id}.{column}."
                logger.info(
                    "integrity_clean_drop_completed",
                    dataset_id=dataset_id,
                    column=column,
                    orphan_count=orphan_count,
                    rows_after=rows_after,
                )
            else:
                # Default behavior for FLAG: add a flag column
                flag_col = f"{column}_is_orphan"
                df_cleaned[flag_col] = is_orphan
                rows_after = rows_before
                msg = f"Flagged {orphan_count} orphan key rows in {dataset_id}.{column} under '{flag_col}'."
                logger.info(
                    "integrity_clean_flag_completed",
                    dataset_id=dataset_id,
                    column=column,
                    orphan_count=orphan_count,
                    flag_column=flag_col,
                )

            result = ExecutorResult(
                success=True,
                message=msg,
                records_before=rows_before,
                records_after=rows_after,
                details={"orphan_count": orphan_count, "action": action},
            )
            return df_cleaned, result

        except Exception as exc:
            msg = f"Integrity cleaning failed for {dataset_id}.{column}: {exc}"
            logger.exception("integrity_clean_failed", dataset_id=dataset_id, column=column)
            return df, ExecutorResult(
                success=False,
                message=msg,
                records_before=rows_before,
                records_after=rows_before,
                error_message=msg,
            )
