"""Null value validation with per-column reporting."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0


class NullValidator(BaseValidator):
    """Counts and reports null values per column."""

    name = "nulls"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Analyze null counts and percentages for each column.

        Args:
            df: Dataset to validate.
            schema: Schema with nullable column metadata.
            **context: Unused.

        Returns:
            ValidationResult: Null analysis outcome.
        """
        row_count = len(df)
        nullable_columns = {col.name for col in schema.columns if col.nullable}
        per_column: dict[str, dict[str, float | int]] = {}
        violations: list[str] = []

        for column in df.columns:
            null_count = int(df[column].isna().sum())
            null_pct = round((null_count / row_count) * 100, 2) if row_count else 0.0
            per_column[column] = {"count": null_count, "percentage": null_pct}
            if column not in nullable_columns and null_count > 0:
                violations.append(column)

        non_nullable_cols = [
            c.name for c in schema.columns if not c.nullable and c.name in df.columns
        ]
        if non_nullable_cols:
            violation_cells = sum(
                int(df[col].isna().sum()) for col in non_nullable_cols
            )
            total_cells = row_count * len(non_nullable_cols)
            violation_rate = violation_cells / total_cells if total_cells else 0.0
        else:
            violation_rate = 0.0

        score = max(0.0, 100.0 * (1.0 - violation_rate))
        passed = score >= PASS_THRESHOLD and not violations

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "row_count": row_count,
                "per_column": per_column,
                "non_nullable_violations": violations,
            },
            duration_seconds=0.0,
        )
