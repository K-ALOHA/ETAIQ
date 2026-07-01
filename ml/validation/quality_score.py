"""Weighted quality score calculation across validation categories."""

from __future__ import annotations

from ml.validation.models import ValidationResult

CATEGORY_WEIGHTS: dict[str, float] = {
    "schema": 0.30,
    "nulls": 0.20,
    "duplicates": 0.15,
    "gps": 0.15,
    "foreign_key": 0.10,
    "target": 0.10,
}

# Timestamp checks contribute to the schema weight bucket.
_VALIDATOR_TO_CATEGORY: dict[str, str] = {
    "schema": "schema",
    "timestamp": "schema",
    "nulls": "nulls",
    "duplicates": "duplicates",
    "gps": "gps",
    "foreign_key": "foreign_key",
    "target": "target",
}


class QualityScoreCalculator:
    """Computes a weighted quality score from validation results."""

    def calculate(
        self, results: list[ValidationResult]
    ) -> tuple[float, dict[str, float]]:
        """Aggregate per-category scores into an overall 0–100 quality score.

        Args:
            results: Individual validation results from the engine.

        Returns:
            tuple[float, dict[str, float]]: Overall score and per-category averages.
        """
        category_scores: dict[str, list[float]] = {key: [] for key in CATEGORY_WEIGHTS}

        for result in results:
            category = _VALIDATOR_TO_CATEGORY.get(result.validator_name)
            if category:
                category_scores[category].append(result.score)

        component_averages: dict[str, float] = {}
        weighted_sum = 0.0
        weight_applied = 0.0

        for category, weight in CATEGORY_WEIGHTS.items():
            scores = category_scores.get(category, [])
            if not scores:
                continue
            average = sum(scores) / len(scores)
            component_averages[category] = round(average, 2)
            weighted_sum += average * weight
            weight_applied += weight

        overall = round(weighted_sum / weight_applied, 2) if weight_applied else 0.0
        return overall, component_averages

    @staticmethod
    def to_report_dict(
        overall_score: float, component_scores: dict[str, float]
    ) -> dict:
        """Build the quality score JSON payload.

        Args:
            overall_score: Weighted overall quality score.
            component_scores: Per-category average scores.

        Returns:
            dict: Serializable quality score report.
        """
        return {
            "overall_score": overall_score,
            "scale": "0-100",
            "weights": {k: v for k, v in CATEGORY_WEIGHTS.items()},
            "component_scores": component_scores,
            "interpretation": _interpret_score(overall_score),
        }


def _interpret_score(score: float) -> str:
    """Return a human-readable quality band for a numeric score.

    Args:
        score: Quality score between 0 and 100.

    Returns:
        str: Quality band label.
    """
    if score >= 90:
        return "excellent"
    if score >= 80:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "poor"
    return "critical"
