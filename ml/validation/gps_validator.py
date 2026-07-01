"""Geographic coordinate validation for latitude and longitude columns."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0

LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)


class GpsValidator(BaseValidator):
    """Validates latitude and longitude values are within valid ranges."""

    name = "gps"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Check GPS columns for range violations and non-numeric values.

        Args:
            df: Dataset to validate.
            schema: Schema defining latitude and longitude column names.
            **context: Unused.

        Returns:
            ValidationResult: GPS validation outcome.
        """
        invalid_counts: dict[str, int] = {}
        total_checked = 0

        for lat_col, lon_col in zip(
            schema.latitude_columns, schema.longitude_columns, strict=False
        ):
            for col, valid_range in (
                (lat_col, LATITUDE_RANGE),
                (lon_col, LONGITUDE_RANGE),
            ):
                if col not in df.columns:
                    continue
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = numeric.dropna()
                total_checked += len(non_null)
                low, high = valid_range
                invalid = int(((non_null < low) | (non_null > high)).sum())
                invalid_counts[col] = invalid

        invalid_total = sum(invalid_counts.values())
        invalid_rate = invalid_total / total_checked if total_checked else 0.0
        score = max(0.0, 100.0 * (1.0 - invalid_rate))
        passed = score >= PASS_THRESHOLD and invalid_total == 0

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "latitude_columns": list(schema.latitude_columns),
                "longitude_columns": list(schema.longitude_columns),
                "invalid_counts": invalid_counts,
                "values_checked": total_checked,
                "valid_latitude_range": list(LATITUDE_RANGE),
                "valid_longitude_range": list(LONGITUDE_RANGE),
            },
            duration_seconds=0.0,
        )
