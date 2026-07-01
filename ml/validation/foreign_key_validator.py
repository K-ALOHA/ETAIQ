"""Foreign key referential integrity validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.validation.models import ValidationResult
from ml.validation.schemas import DatasetSchema
from ml.validation.validator import BaseValidator

PASS_THRESHOLD = 80.0

FOREIGN_KEY_MAP: dict[str, tuple[str, str]] = {
    "restaurant_id": ("restaurant_id", "restaurants"),
    "rider_id": ("rider_id", "riders"),
}


class ForeignKeyValidator(BaseValidator):
    """Validates orders reference existing restaurants and riders."""

    name = "foreign_key"

    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: Any
    ) -> ValidationResult:
        """Check foreign key columns against reference dataset IDs.

        Args:
            df: Orders dataset to validate.
            schema: Orders schema definition.
            **context: Must include ``reference_ids`` mapping dataset names to ID sets.

        Returns:
            ValidationResult: Foreign key validation outcome.
        """
        reference_ids: dict[str, set[str]] = context.get("reference_ids", {})
        orphan_details: dict[str, dict[str, int | list[str]]] = {}
        total_checked = 0
        total_orphans = 0

        for fk_column, (ref_id_col, ref_dataset) in FOREIGN_KEY_MAP.items():
            if fk_column not in df.columns:
                continue

            ref_set = reference_ids.get(ref_dataset, set())
            values = df[fk_column].dropna().astype(str)
            total_checked += len(values)

            if not ref_set:
                orphan_details[fk_column] = {
                    "orphan_count": len(values),
                    "reference_dataset_missing": True,
                    "sample_orphan_values": values.head(5).tolist(),
                }
                total_orphans += len(values)
                continue

            orphans = values[~values.isin(ref_set)]
            orphan_count = int(len(orphans))
            total_orphans += orphan_count
            orphan_details[fk_column] = {
                "orphan_count": orphan_count,
                "reference_dataset": ref_dataset,
                "reference_id_column": ref_id_col,
                "sample_orphan_values": orphans.head(10).astype(str).tolist(),
            }

        orphan_rate = total_orphans / total_checked if total_checked else 0.0
        score = max(0.0, 100.0 * (1.0 - orphan_rate))
        passed = score >= PASS_THRESHOLD and total_orphans == 0

        return ValidationResult(
            validator_name=self.name,
            dataset_name=schema.name,
            passed=passed,
            score=score,
            details={
                "foreign_keys_checked": list(FOREIGN_KEY_MAP.keys()),
                "total_references_checked": total_checked,
                "total_orphans": total_orphans,
                "per_column": orphan_details,
            },
            duration_seconds=0.0,
        )
