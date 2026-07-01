"""Duplicate row and duplicate identifier validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0


class DuplicateValidator(BaseValidator):
    """Detects exact duplicate rows and duplicate primary keys."""

    name = "duplicates"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Find exact duplicates and duplicate ID values.

        Args:
            df: Dataset to validate.
            schema: Schema containing the ID column name.
            **context: Unused.

        Returns:
            ValidationResult: Duplicate analysis outcome.
        """
        row_count = len(df)
        exact_duplicate_count = int(df.duplicated(keep=False).sum())
        exact_duplicate_rows = int(df.duplicated().sum())

        duplicate_id_count = 0
        duplicate_ids: list[str] = []
        if schema.id_column in df.columns:
            id_series = df[schema.id_column].dropna().astype(str)
            duplicated_mask = id_series.duplicated(keep=False)
            duplicate_id_count = int(duplicated_mask.sum())
            if duplicate_id_count:
                duplicate_ids = (
                    id_series[duplicated_mask]
                    .value_counts()
                    .head(10)
                    .index.astype(str)
                    .tolist()
                )

        exact_rate = exact_duplicate_rows / row_count if row_count else 0.0
        id_rate = duplicate_id_count / row_count if row_count else 0.0
        combined_rate = min(1.0, exact_rate + id_rate)
        score = max(0.0, 100.0 * (1.0 - combined_rate))
        passed = score >= PASS_THRESHOLD and duplicate_id_count == 0

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "row_count": row_count,
                "exact_duplicate_rows": exact_duplicate_rows,
                "exact_duplicate_cells_flagged": exact_duplicate_count,
                "duplicate_id_count": duplicate_id_count,
                "sample_duplicate_ids": duplicate_ids,
                "id_column": schema.id_column,
            },
            duration_seconds=0.0,
        )
