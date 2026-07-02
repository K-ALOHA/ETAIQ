"""Registry manager for feature engineering components."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .config import FeatureEngineeringConfig
from .logging_config import FeatureEngineeringLogger
from .models import FeatureMetadata, FeatureRegistry
from .utils import ensure_directory


class FeatureRegistryManager:
    """Manager for feature metadata registration and export."""

    TARGET_FEATURE_NAME = "actual_delivery_time_min"
    FEATURE_CATEGORIES = [
        "Identifier",
        "Target",
        "Numerical",
        "Categorical",
        "Datetime",
        "GPS",
        "Boolean",
        "Unknown",
    ]
    RECOMMENDED_ACTIONS = {
        "Identifier": "Drop",
        "Target": "Prediction Target",
        "Numerical": "Scale",
        "Categorical": "Encode",
        "Datetime": "Extract Features",
        "GPS": "Distance Engineering",
        "Boolean": "Keep",
        "Unknown": "Review",
    }
    IDENTIFIER_PATTERN = re.compile(r"(^|_)id($|_)", re.IGNORECASE)
    GPS_PATTERNS = ("lat", "latitude", "lon", "longitude")
    TIMESTAMP_PATTERNS = ("timestamp", "datetime")

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.registry = FeatureRegistry()

    def initialize(self) -> FeatureRegistry:
        """Initialize the feature registry."""
        self.registry = FeatureRegistry()
        self.logger.info("Registry ready")
        return self.registry

    def register_feature(self, feature: FeatureMetadata) -> FeatureMetadata:
        """Register a feature metadata entry."""
        self.logger.info("Registering feature", feature_name=feature.name, feature_type=feature.feature_type)
        self.registry.features.append(feature)
        return feature

    def list_features(self) -> list[FeatureMetadata]:
        """Return all registered features."""
        self.logger.info("Listing registered features")
        return list(self.registry.features)

    def inspect_features(self, training_df: Any) -> FeatureRegistry:
        """Inspect the merged training dataframe and classify every feature."""
        self.initialize()
        self.logger.info("Inspecting merged dataframe", columns=len(training_df.columns))

        for column in training_df.columns:
            series = training_df[column]
            category = self._classify_feature(column, series)
            recommended_action = self.RECOMMENDED_ACTIONS[category]
            metadata = FeatureMetadata(
                name=column,
                source="merged_training_df",
                feature_type=category,
                original_dtype=str(series.dtype),
                recommended_action=recommended_action,
                description=recommended_action,
            )
            self.register_feature(metadata)

        self._print_summary()
        return self.registry

    def _classify_feature(self, column: str, series: Any) -> str:
        lower_name = column.lower()
        if lower_name == self.TARGET_FEATURE_NAME:
            return "Target"
        if self._is_identifier(column):
            return "Identifier"
        if self._is_gps(column):
            return "GPS"
        if self._is_datetime(column, series):
            return "Datetime"
        if self._is_boolean(series):
            return "Boolean"
        if self._is_numeric(series):
            return "Numerical"
        if self._is_categorical(series):
            return "Categorical"
        return "Unknown"

    def _is_identifier(self, column: str) -> bool:
        lower_name = column.lower()
        if lower_name == "id":
            return True
        if lower_name.endswith("_id"):
            return True
        return bool(self.IDENTIFIER_PATTERN.search(lower_name))

    def _is_gps(self, column: str) -> bool:
        lower_name = column.lower()
        return any(token in lower_name for token in self.GPS_PATTERNS)

    def _is_datetime(self, column: str, series: Any) -> bool:
        lower_name = column.lower()
        if any(token in lower_name for token in self.TIMESTAMP_PATTERNS):
            return True
        try:
            return series.dtype.kind == "M"
        except AttributeError:
            return False

    def _is_boolean(self, series: Any) -> bool:
        try:
            return series.dtype == bool or series.dtype.kind == "b"
        except AttributeError:
            return False

    def _is_numeric(self, series: Any) -> bool:
        try:
            return series.dtype.kind in {"i", "u", "f"}
        except AttributeError:
            return False

    def _is_categorical(self, series: Any) -> bool:
        try:
            return (
                isinstance(series.dtype, pd.CategoricalDtype)
                or is_string_dtype(series.dtype)
                or is_object_dtype(series.dtype)
            )
        except AttributeError:
            return False

    def _print_summary(self) -> None:
        counts = {category: 0 for category in self.FEATURE_CATEGORIES}
        for feature in self.registry.features:
            counts[feature.feature_type] += 1

        print("=" * 40)
        print("Feature Registry Summary")
        print("=" * 40)
        for category in self.FEATURE_CATEGORIES:
            label = category + ("s" if not category.endswith("s") else "")
            print(f"{label} : {counts[category]}")
        print("=" * 40)

    def export_registry(self) -> str:
        """Export the registry to disk as a CSV file."""
        if not self.registry.features:
            raise ValueError("No features registered to export.")

        ensure_directory(self.config.feature_registry_output_path.parent)
        output_path = Path(self.config.feature_registry_output_path)
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["feature_name", "feature_type", "original_dtype", "recommended_action"])
            for feature in self.registry.features:
                writer.writerow([
                    feature.name,
                    feature.feature_type,
                    feature.original_dtype,
                    feature.recommended_action,
                ])

        self.logger.info("Feature registry exported", path=str(output_path))
        return str(output_path)
