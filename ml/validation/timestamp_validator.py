"""Timestamp parsing and consistency validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0


class TimestampValidator(BaseValidator):
    """Detects timestamp columns that fail parsing."""

    name = "timestamp"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Validate configured timestamp columns parse correctly.

        Args:
            df: Dataset to validate.
            schema: Schema listing timestamp column names.
            **context: Unused.

        Returns:
            ValidationResult: Timestamp validation outcome.
        """
        per_column: dict[str, dict[str, int | float]] = {}
        total_non_null = 0
        total_invalid = 0

        for column in schema.timestamp_columns:
            if column not in df.columns:
                continue
            series = df[column].dropna()
            non_null_count = len(series)
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            invalid_count = int(parsed.isna().sum())
            total_non_null += non_null_count
            total_invalid += invalid_count
            per_column[column] = {
                "non_null_count": non_null_count,
                "invalid_count": invalid_count,
                "invalid_percentage": round((invalid_count / non_null_count) * 100, 2)
                if non_null_count
                else 0.0,
            }

        invalid_rate = total_invalid / total_non_null if total_non_null else 0.0
        score = max(0.0, 100.0 * (1.0 - invalid_rate))
        passed = score >= PASS_THRESHOLD and total_invalid == 0

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "timestamp_columns": list(schema.timestamp_columns),
                "per_column": per_column,
                "total_invalid": total_invalid,
            },
            duration_seconds=0.0,
        )
