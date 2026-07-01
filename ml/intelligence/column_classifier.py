"""Heuristic column role classification."""

from __future__ import annotations

import time

from ml.intelligence.config import (
    DEFAULT_CONFIG,
    ID_NAME_PATTERN,
    METADATA_NAME_PATTERN,
    PII_NAME_PATTERN,
    TARGET_NAME_PATTERN,
    TIME_NAME_PATTERN,
    IntelligenceConfig,
)
from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import ColumnProfile, ColumnRole, DatasetProfile

logger = get_logger(__name__)


class ColumnClassifier:
    """Classifies columns using statistical and naming heuristics."""

    def __init__(self, config: IntelligenceConfig = DEFAULT_CONFIG) -> None:
        """Initialize the classifier.

        Args:
            config: Intelligence configuration.
        """
        self._config = config

    def classify_profiles(self, profiles: list[DatasetProfile]) -> list[DatasetProfile]:
        """Assign role labels to every column profile."""
        logger.info("column_classification_start", datasets=len(profiles))
        start = time.perf_counter()

        for profile in profiles:
            for column in profile.columns:
                column.roles = self._classify_column(column, profile)

        logger.info(
            "column_classification_end",
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return profiles

    def _classify_column(
        self, column: ColumnProfile, profile: DatasetProfile
    ) -> list[str]:
        """Infer role labels for a single column."""
        roles: list[str] = []
        name = column.name.lower()
        uniqueness = column.unique_percentage / 100 if profile.row_count else 0.0
        cfg = self._config

        if (
            ID_NAME_PATTERN.search(name)
            or uniqueness >= cfg.identifier_uniqueness_ratio
        ):
            roles.append(ColumnRole.IDENTIFIER.value)
        if (
            TIME_NAME_PATTERN.search(name) or column.is_datetime
        ) and not column.is_numeric:
            roles.append(ColumnRole.TIMESTAMP.value)
        if PII_NAME_PATTERN.search(name):
            roles.append(ColumnRole.PII.value)
        if METADATA_NAME_PATTERN.search(name):
            roles.append(ColumnRole.METADATA.value)
        if column.is_boolean:
            roles.append(ColumnRole.BOOLEAN_FEATURE.value)
        if column.is_numeric and ColumnRole.IDENTIFIER.value not in roles:
            roles.append(ColumnRole.NUMERIC_FEATURE.value)
        if column.is_categorical:
            roles.append(ColumnRole.CATEGORICAL_FEATURE.value)
        if column.is_text:
            roles.append(ColumnRole.TEXT.value)
        if column.is_high_cardinality and ColumnRole.IDENTIFIER.value not in roles:
            roles.append(ColumnRole.HIGH_CARDINALITY.value)
        if TARGET_NAME_PATTERN.search(name) and column.is_numeric:
            roles.append(ColumnRole.POTENTIAL_TARGET.value)
        if column.null_percentage >= 99 or (
            column.unique_count <= 1 and profile.row_count > 1
        ):
            roles.append(ColumnRole.DISTRACTOR.value)

        if not roles:
            roles.append(ColumnRole.UNKNOWN.value)
        return list(dict.fromkeys(roles))

    @staticmethod
    def mark_foreign_keys(
        profiles: list[DatasetProfile],
        foreign_key_map: dict[tuple[str, str], tuple[str, str]],
    ) -> None:
        """Annotate columns detected as foreign keys."""
        for profile in profiles:
            for column in profile.columns:
                key = (profile.dataset_id, column.name)
                if key in foreign_key_map:
                    if ColumnRole.FOREIGN_KEY.value not in column.roles:
                        column.roles.append(ColumnRole.FOREIGN_KEY.value)
