"""Schema validation: columns, types, and structural integrity."""

from __future__ import annotations

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import ColumnDtype, ColumnSpec, DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0


def _check_column_dtype(series: pd.Series, spec: ColumnSpec) -> list[str]:
    """Return dtype mismatch messages for a single column.

    Args:
        series: Column data to inspect.
        spec: Expected column specification.

    Returns:
        list[str]: Human-readable mismatch descriptions.
    """
    issues: list[str] = []
    non_null = series.dropna()
    if non_null.empty:
        return issues

    if spec.dtype == ColumnDtype.INTEGER:
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.isna().any():
            issues.append(f"{spec.name}: non-integer values detected")
        elif not ((numeric % 1) == 0).all():
            issues.append(f"{spec.name}: fractional values in integer column")
    elif spec.dtype == ColumnDtype.FLOAT:
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.isna().any():
            issues.append(f"{spec.name}: non-numeric values detected")
    elif spec.dtype == ColumnDtype.BOOLEAN:
        allowed = {True, False, "true", "false", "True", "False", "1", "0", 1, 0}
        invalid = non_null[~non_null.isin(allowed)]
        if not invalid.empty:
            issues.append(f"{spec.name}: invalid boolean values detected")
    elif spec.dtype == ColumnDtype.DATETIME:
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
        if parsed.isna().any():
            issues.append(f"{spec.name}: unparseable datetime values detected")
    return issues


class SchemaValidator(BaseValidator):
    """Validates required columns, extras, and logical data types."""

    name = "schema"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: object
    ) -> ValidationResult:
        """Check schema conformance for a dataset.

        Args:
            df: Dataset to validate.
            schema: Expected schema definition.
            **context: Unused.

        Returns:
            ValidationResult: Schema validation outcome.
        """
        expected = {col.name for col in schema.columns if col.required}
        optional = {col.name for col in schema.columns if not col.required}
        all_expected = expected | optional
        actual = set(df.columns)

        missing = sorted(expected - actual)
        extra = sorted(actual - all_expected)

        dtype_issues: dict[str, list[str]] = {}
        for spec in schema.columns:
            if spec.name not in df.columns:
                continue
            issues = _check_column_dtype(df[spec.name], spec)
            if issues:
                dtype_issues[spec.name] = issues

        penalty = len(missing) * 20 + len(extra) * 5 + len(dtype_issues) * 10
        score = max(0.0, 100.0 - penalty)
        passed = score >= PASS_THRESHOLD and not missing

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "required_columns": sorted(expected),
                "missing_columns": missing,
                "extra_columns": extra,
                "dtype_issues": dtype_issues,
                "column_count": len(df.columns),
            },
            duration_seconds=0.0,
        )
