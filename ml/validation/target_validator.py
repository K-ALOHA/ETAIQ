"""Target variable validation for delivery time and related metrics."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0

MAX_DELIVERY_MINUTES = 24 * 60  # 24 hours


class TargetValidator(BaseValidator):
    """Validates regression target columns for impossible values."""

    name = "target"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Detect negative and impossibly large delivery time values.

        Args:
            df: Dataset to validate.
            schema: Schema listing target column names.
            **context: Unused.

        Returns:
            ValidationResult: Target validation outcome.
        """
        per_column: dict[str, dict[str, int | float]] = {}
        total_checked = 0
        total_invalid = 0

        for column in schema.target_columns:
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            non_null = numeric.dropna()
            checked = len(non_null)
            negative = int((non_null < 0).sum())
            too_large = int((non_null > MAX_DELIVERY_MINUTES).sum())
            non_numeric = int(numeric.isna().sum()) - int(df[column].isna().sum())
            invalid = negative + too_large + max(0, non_numeric)
            total_checked += checked
            total_invalid += invalid
            per_column[column] = {
                "values_checked": checked,
                "negative_count": negative,
                "impossible_high_count": too_large,
                "non_numeric_count": max(0, non_numeric),
                "max_allowed_minutes": MAX_DELIVERY_MINUTES,
            }

        invalid_rate = total_invalid / total_checked if total_checked else 0.0
        score = max(0.0, 100.0 * (1.0 - invalid_rate))
        passed = score >= PASS_THRESHOLD and total_invalid == 0

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "target_columns": list(schema.target_columns),
                "per_column": per_column,
                "total_invalid": total_invalid,
            },
            duration_seconds=0.0,
        )
