"""Automatic prediction target candidate detection with confidence tiers."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ml.intelligence.config import (
    COORDINATE_NAME_PATTERN,
    DEFAULT_CONFIG,
    POST_EVENT_TARGET_PATTERN,
    PRE_EVENT_FEATURE_PATTERN,
    TARGET_NAME_PATTERN,
    IntelligenceConfig,
)
from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import ColumnProfile, DatasetProfile, TargetCandidate

logger = get_logger(__name__)


@dataclass
class _TargetScore:
    """Internal scoring result for a target candidate."""

    confidence: float
    target_type: str
    evidence: list[str]
    explanation: str


class TargetDetector:
    """Ranks columns that are likely prediction targets using confidence tiers."""

    def __init__(self, config: IntelligenceConfig = DEFAULT_CONFIG) -> None:
        """Initialize the detector.

        Args:
            config: Intelligence configuration.
        """
        self._config = config

    def detect(self, profiles: list[DatasetProfile]) -> list[TargetCandidate]:
        """Identify and rank high-confidence target candidates only.

        Returns candidates grouped internally by tier: strong, possible, weak.
        Only columns above the weak threshold are included.
        """
        logger.info("target_detection_start", datasets=len(profiles))
        start = time.perf_counter()

        scored: list[tuple[float, TargetCandidate]] = []
        for profile in profiles:
            for column in profile.columns:
                result = self._score_column(column)
                if (
                    result is None
                    or result.confidence < self._config.target_weak_threshold
                ):
                    continue
                tier = self._tier_for_confidence(result.confidence)
                scored.append(
                    (
                        result.confidence,
                        TargetCandidate(
                            dataset_id=profile.dataset_id,
                            column=column.name,
                            rank=0,
                            score=round(result.confidence, 4),
                            confidence=round(result.confidence, 4),
                            tier=tier,
                            target_type=result.target_type,
                            rationale=result.explanation,
                            explanation=result.explanation,
                            evidence=result.evidence,
                        ),
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [
            TargetCandidate(
                dataset_id=item.dataset_id,
                column=item.column,
                rank=rank,
                score=item.score,
                confidence=item.confidence,
                tier=item.tier,
                target_type=item.target_type,
                rationale=item.rationale,
                explanation=item.explanation,
                evidence=item.evidence,
            )
            for rank, (_, item) in enumerate(scored, start=1)
        ]

        logger.info(
            "target_detection_end",
            candidates=len(candidates),
            strong=sum(1 for c in candidates if c.tier == "strong"),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return candidates

    def _score_column(self, column: ColumnProfile) -> _TargetScore | None:
        """Score a column as a potential target with evidence."""
        cfg = self._config
        name = column.name

        if "foreign_key" in column.roles:
            return None
        if column.is_datetime or (
            "timestamp" in column.roles and not POST_EVENT_TARGET_PATTERN.search(name)
        ):
            return None
        if "metadata" in column.roles:
            return None
        if "distractor" in column.roles and not (
            TARGET_NAME_PATTERN.search(name) or POST_EVENT_TARGET_PATTERN.search(name)
        ):
            return None
        if "identifier" in column.roles and not TARGET_NAME_PATTERN.search(name):
            return None
        if COORDINATE_NAME_PATTERN.search(name):
            return None

        evidence: list[str] = []
        confidence = 0.0
        target_type = "unknown"

        if "potential_target" in column.roles:
            confidence += 0.15
            evidence.append("Classified as potential outcome column.")

        if POST_EVENT_TARGET_PATTERN.search(name):
            confidence += cfg.target_post_event_boost
            evidence.append("Represents post-event outcome.")
        if PRE_EVENT_FEATURE_PATTERN.search(name):
            confidence -= cfg.target_pre_event_penalty
            evidence.append("Available before prediction; better as a feature.")

        if TARGET_NAME_PATTERN.search(name):
            confidence += cfg.target_name_boost
            evidence.append("Column name matches outcome heuristics.")

        if column.is_numeric and column.unique_count > 1:
            confidence += cfg.target_numeric_boost
            evidence.append("Continuous variable with suitable variance.")
            target_type = "regression"
        elif column.is_categorical and 2 <= column.unique_count <= 20:
            if TARGET_NAME_PATTERN.search(name):
                confidence += 0.15
                evidence.append("Moderate cardinality suitable for classification.")
                target_type = "classification"
        elif column.is_boolean and TARGET_NAME_PATTERN.search(name):
            confidence += 0.10
            evidence.append("Binary outcome column.")
            target_type = "classification"

        if column.null_percentage > 30:
            confidence -= 0.15
            evidence.append("High null rate reduces target suitability.")

        confidence = max(0.0, min(confidence, 1.0))
        if confidence < cfg.target_weak_threshold:
            return None

        explanation = self._build_explanation(column, target_type, evidence)
        return _TargetScore(
            confidence=confidence,
            target_type=target_type,
            evidence=evidence,
            explanation=explanation,
        )

    def _tier_for_confidence(self, confidence: float) -> str:
        """Map a confidence score to a tier label."""
        cfg = self._config
        if confidence >= cfg.target_strong_threshold:
            return "strong"
        if confidence >= cfg.target_possible_threshold:
            return "possible"
        return "weak"

    @staticmethod
    def _build_explanation(
        column: ColumnProfile,
        target_type: str,
        evidence: list[str],
    ) -> str:
        """Build a human-readable explanation string."""
        parts = []
        if column.is_numeric:
            parts.append("Continuous variable.")
        if target_type == "regression":
            parts.append("Suitable regression target.")
        elif target_type == "classification":
            parts.append("Suitable classification target.")
        parts.extend(evidence)
        return " ".join(parts)
