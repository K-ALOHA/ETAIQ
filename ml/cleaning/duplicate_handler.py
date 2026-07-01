"""De-duplication handler to remove duplicate rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import ExecutorResult

logger = get_logger(__name__)


class DuplicateHandler:
    """Removes duplicate rows or subset duplicates from a DataFrame."""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        column: str | None = None,
        **details: Any,
    ) -> tuple[pd.DataFrame, ExecutorResult]:
        """Drop exact duplicate rows.

        If a column is specified, drops duplicates based on that column (or primary keys).
        Otherwise, drops exact duplicate rows.
        """
        rows_before = len(df)
        subset = [column] if column else None

        # Perform de-duplication
        df_cleaned = df.drop_duplicates(subset=subset, keep="first").copy()
        rows_after = len(df_cleaned)
        dropped = rows_before - rows_after

        msg = f"Removed {dropped} duplicate rows from {dataset_id}."
        logger.info(
            "remove_duplicates_completed",
            dataset_id=dataset_id,
            rows_before=rows_before,
            rows_after=rows_after,
            dropped=dropped,
        )

        result = ExecutorResult(
            success=True,
            message=msg,
            records_before=rows_before,
            records_after=rows_after,
            details={"dropped_count": dropped, "subset": subset},
        )
        return df_cleaned, result
