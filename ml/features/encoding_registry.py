"""Registry and export helpers for the encoding framework."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import FeatureEngineeringConfig
from .logging_config import FeatureEngineeringLogger
from .models import FeatureMetadata, FeatureRegistry
from .utils import ensure_directory


@dataclass
class EncodingPlanEntry:
    """A single entry in the encoding plan."""

    feature_name: str
    feature_type: str
    encoding_strategy: str


@dataclass
class EncodingPlan:
    """A list of encoding plan entries."""

    entries: list[EncodingPlanEntry] = field(default_factory=list)

    def add_entry(self, feature_name: str, feature_type: str, encoding_strategy: str) -> None:
        self.entries.append(EncodingPlanEntry(feature_name, feature_type, encoding_strategy))

    def __len__(self) -> int:
        return len(self.entries)


class EncodingRegistry:
    """Create an encoding plan from the feature registry."""

    ORDINAL_CATEGORICALS = {
        "rider_experience_level",
        "restaurant_quality_tier",
    }
    NO_ENCODING = "No Encoding"
    ONEHOT_ENCODING = "OneHot Encoding"
    ORDINAL_ENCODING = "Ordinal Encoding"
    SKIPPED_HIGH_CARDINALITY = "Skipped (High Cardinality)"

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.plan = EncodingPlan()

    def create_plan(
        self,
        feature_registry: FeatureRegistry | list[FeatureMetadata],
        X_train: pd.DataFrame | None = None,
    ) -> EncodingPlan:
        """Create an encoding plan based on feature metadata."""
        self.plan = EncodingPlan()
        features = feature_registry.features if isinstance(feature_registry, FeatureRegistry) else feature_registry
        for feature in features:
            encoding_strategy = self._select_strategy(feature.feature_type, feature.name)
            if encoding_strategy == self.ONEHOT_ENCODING and X_train is not None:
                if self._is_high_cardinality(feature.name, X_train):
                    encoding_strategy = self.SKIPPED_HIGH_CARDINALITY
                    self.logger.info(
                        "High-cardinality categorical skipped",
                        feature_name=feature.name,
                        unique_values=self._get_unique_value_count(feature.name, X_train),
                        ratio=self._get_cardinality_ratio(feature.name, X_train),
                    )
            self.plan.add_entry(feature.name, feature.feature_type, encoding_strategy)

        self.logger.info("Encoding registry created", entries=len(self.plan))
        return self.plan

    def _select_strategy(self, feature_type: str, feature_name: str) -> str:
        if feature_type in {"Identifier", "Target", "Numerical", "GPS", "Boolean"}:
            return self.NO_ENCODING
        if feature_type == "Categorical":
            return self.ORDINAL_ENCODING if self._is_ordinal(feature_name) else self.ONEHOT_ENCODING
        return self.NO_ENCODING

    def _is_ordinal(self, feature_name: str) -> bool:
        return feature_name.lower() in self.ORDINAL_CATEGORICALS

    def _get_unique_value_count(self, feature_name: str, X_train: pd.DataFrame) -> int:
        if feature_name not in X_train.columns:
            return 0
        return int(X_train[feature_name].nunique(dropna=False))

    def _get_cardinality_ratio(self, feature_name: str, X_train: pd.DataFrame) -> float:
        total_rows = len(X_train)
        if total_rows == 0:
            return 0.0
        return self._get_unique_value_count(feature_name, X_train) / total_rows

    def _is_high_cardinality(self, feature_name: str, X_train: pd.DataFrame) -> bool:
        if feature_name not in X_train.columns:
            return False
        unique_values = self._get_unique_value_count(feature_name, X_train)
        return unique_values > 100 or self._get_cardinality_ratio(feature_name, X_train) > 0.10

    def export_plan(self, plan: EncodingPlan) -> str:
        """Export the encoding plan to a CSV file."""
        output_path = Path(self.config.project_root) / "ml" / "data" / "features" / "encoding_plan.csv"
        ensure_directory(output_path.parent)

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["feature_name", "feature_type", "encoding_strategy"])
            for entry in plan.entries:
                writer.writerow([entry.feature_name, entry.feature_type, entry.encoding_strategy])

        self.logger.info("Encoding plan exported", path=str(output_path))
        return str(output_path)
