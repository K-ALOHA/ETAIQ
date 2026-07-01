"""Feature usefulness and engineering recommendations with confidence tiers."""

from __future__ import annotations

import time

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import ColumnProfile, DatasetProfile, FeatureCandidate

logger = get_logger(__name__)


class FeatureRecommender:
    """Recommends feature treatment for every column with explicit classifications."""

    def __init__(self, config: IntelligenceConfig = DEFAULT_CONFIG) -> None:
        """Initialize the recommender.

        Args:
            config: Intelligence configuration.
        """
        self._config = config

    def recommend(self, profiles: list[DatasetProfile]) -> list[FeatureCandidate]:
        """Generate ranked feature recommendations for all columns."""
        logger.info("feature_recommendation_start", datasets=len(profiles))
        start = time.perf_counter()
        candidates = [
            self._recommend_column(profile.dataset_id, column)
            for profile in profiles
            for column in profile.columns
        ]
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        logger.info(
            "feature_recommendation_end",
            candidates=len(candidates),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return candidates

    def _recommend_column(
        self, dataset_id: str, column: ColumnProfile
    ) -> FeatureCandidate:
        """Recommend treatment for a single column."""
        roles = set(column.roles)

        if "identifier" in roles:
            return self._build(
                dataset_id,
                column,
                classification="identifier",
                recommendation="ignore",
                confidence=0.95,
                reason="Unique identifier. No predictive value.",
                action="Drop before training.",
            )
        if "foreign_key" in roles:
            return self._build(
                dataset_id,
                column,
                classification="identifier",
                recommendation="ignore",
                confidence=0.90,
                reason="Foreign key used for joins, not direct modeling.",
                action="Use for enrichment joins only.",
            )
        if "pii" in roles:
            return self._build(
                dataset_id,
                column,
                classification="pii",
                recommendation="ignore",
                confidence=0.98,
                reason="Personally identifiable information.",
                action="Exclude for privacy compliance.",
            )
        if "metadata" in roles:
            return self._build(
                dataset_id,
                column,
                classification="metadata",
                recommendation="ignore",
                confidence=0.92,
                reason="Operational metadata column.",
                action="Drop before training.",
            )
        if "potential_leakage" in roles:
            return self._build(
                dataset_id,
                column,
                classification="leakage",
                recommendation="ignore",
                confidence=0.96,
                reason="Suspected target leakage.",
                action="Do not use as a feature.",
            )
        if "potential_target" in roles:
            return self._build(
                dataset_id,
                column,
                classification="target",
                recommendation="ignore",
                confidence=0.94,
                reason="Identified as prediction target, not input feature.",
                action="Use as label only.",
            )
        if "distractor" in roles:
            return self._build(
                dataset_id,
                column,
                classification="distractor",
                recommendation="ignore",
                confidence=0.88,
                reason="Near-constant or empty column.",
                action="Drop before training.",
            )

        if column.null_percentage > self._config.feature_high_null_penalty * 100:
            return self._build(
                dataset_id,
                column,
                classification="optional_feature",
                recommendation="weak_feature",
                confidence=0.35,
                reason="Very high null rate reduces predictive utility.",
                action="Impute or exclude unless business-critical.",
            )

        if column.is_datetime:
            return self._build(
                dataset_id,
                column,
                classification="derived_feature",
                recommendation="derived_feature_candidate",
                confidence=0.88,
                reason="Datetime should be decomposed into calendar features.",
                action="Extract year, month, day-of-week, hour.",
                encoding="none",
                scaling="none",
                engineering=[
                    "extract_year",
                    "extract_month",
                    "extract_day_of_week",
                    "extract_hour",
                ],
            )
        if column.is_text:
            return self._build(
                dataset_id,
                column,
                classification="derived_feature",
                recommendation="derived_feature_candidate",
                confidence=0.62,
                reason="Free-text requires NLP feature extraction.",
                action="Derive length, token count, or embeddings.",
                engineering=["text_length", "token_count", "embedding_candidate"],
            )
        if column.is_numeric:
            confidence = 0.82
            classification = "recommended_feature"
            if column.unique_percentage > 90:
                confidence = 0.55
                classification = "optional_feature"
            return self._build(
                dataset_id,
                column,
                classification=classification,
                recommendation="useful_feature",
                confidence=confidence,
                reason="Numeric column suitable for scaling and tree/linear models.",
                action="Include with standard scaling.",
                encoding="none",
                scaling="standard_scaler",
            )
        if column.is_boolean:
            return self._build(
                dataset_id,
                column,
                classification="recommended_feature",
                recommendation="useful_feature",
                confidence=0.78,
                reason="Boolean column usable as a binary feature.",
                action="Include directly.",
                encoding="none",
                scaling="none",
            )
        if column.is_categorical:
            encoding = "one_hot" if column.unique_count <= 20 else "target_encoding"
            confidence = 0.80 if column.unique_count <= 20 else 0.58
            classification = (
                "recommended_feature" if confidence >= 0.65 else "optional_feature"
            )
            return self._build(
                dataset_id,
                column,
                classification=classification,
                recommendation="useful_feature",
                confidence=confidence,
                reason="Categorical column suitable for encoding.",
                action=f"Apply {encoding}.",
                encoding=encoding,
                scaling="none",
            )
        if column.is_high_cardinality:
            return self._build(
                dataset_id,
                column,
                classification="optional_feature",
                recommendation="weak_feature",
                confidence=0.48,
                reason="High cardinality may cause overfitting.",
                action="Apply hashing, embedding, or grouping.",
                encoding="hashing_or_embedding",
            )

        return self._build(
            dataset_id,
            column,
            classification="unknown",
            recommendation="weak_feature",
            confidence=0.30,
            reason="Insufficient signal heuristics; inspect manually.",
            action="Review before including.",
        )

    def _build(
        self,
        dataset_id: str,
        column: ColumnProfile,
        classification: str,
        recommendation: str,
        confidence: float,
        reason: str,
        action: str,
        encoding: str | None = None,
        scaling: str | None = None,
        engineering: list[str] | None = None,
    ) -> FeatureCandidate:
        """Construct a feature candidate with unified explainability fields."""
        return FeatureCandidate(
            dataset_id=dataset_id,
            column=column.name,
            classification=classification,
            recommendation=recommendation,
            confidence=round(confidence, 4),
            reason=reason,
            encoding=encoding,
            scaling=scaling,
            engineering=engineering or [],
            rationale=f"{reason} {action}",
        )
