"""Intelligence quality scoring for the Dataset Intelligence Engine."""

from __future__ import annotations

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.models import IntelligenceReport


class IntelligenceScoreCalculator:
    """Computes component and overall intelligence quality scores."""

    def __init__(self, config: IntelligenceConfig = DEFAULT_CONFIG) -> None:
        """Initialize the calculator.

        Args:
            config: Intelligence configuration.
        """
        self._config = config

    def calculate(self, report: IntelligenceReport) -> dict[str, object]:
        """Compute intelligence scores from a completed report.

        Args:
            report: Completed intelligence report.

        Returns:
            dict[str, object]: Intelligence score payload.
        """
        discovery = self._discovery_quality(report)
        relationship = self._relationship_confidence(report)
        schema = self._schema_confidence(report)
        feature = self._feature_confidence(report)
        target = self._target_confidence(report)

        cfg = self._config
        overall = round(
            discovery * cfg.score_discovery_weight
            + relationship * cfg.score_relationship_weight
            + schema * cfg.score_schema_weight
            + feature * cfg.score_feature_weight
            + target * cfg.score_target_weight,
            2,
        )

        return {
            "discovery_quality": round(discovery, 2),
            "relationship_confidence": round(relationship, 2),
            "schema_confidence": round(schema, 2),
            "feature_confidence": round(feature, 2),
            "target_confidence": round(target, 2),
            "overall_intelligence_score": overall,
            "scale": "0-100",
        }

    @staticmethod
    def _discovery_quality(report: IntelligenceReport) -> float:
        """Score dataset discovery completeness."""
        if not report.datasets:
            return 0.0
        profiled = sum(1 for d in report.datasets if d.columns)
        return min(100.0, (profiled / max(len(report.datasets), 1)) * 100)

    @staticmethod
    def _relationship_confidence(report: IntelligenceReport) -> float:
        """Score average relationship detection confidence."""
        if not report.relationships:
            return 50.0
        avg = sum(r.join_confidence for r in report.relationships) / len(
            report.relationships
        )
        return avg * 100

    @staticmethod
    def _schema_confidence(report: IntelligenceReport) -> float:
        """Score schema inference confidence."""
        datasets = report.schema_registry.get("datasets", {})
        if not datasets:
            return 0.0
        typed_columns = 0
        total_columns = 0
        for dataset in datasets.values():
            for col in dataset.get("columns", {}).values():
                total_columns += 1
                if col.get("logical_dtype") not in {None, "unknown"}:
                    typed_columns += 1
        return (typed_columns / max(total_columns, 1)) * 100

    @staticmethod
    def _feature_confidence(report: IntelligenceReport) -> float:
        """Score feature recommendation confidence for actionable features."""
        actionable = [
            f
            for f in report.feature_candidates
            if f.classification not in {"identifier", "metadata", "pii"}
        ]
        if not actionable:
            return 50.0
        avg = sum(f.confidence for f in actionable) / len(actionable)
        return avg * 100

    @staticmethod
    def _target_confidence(report: IntelligenceReport) -> float:
        """Score target detection confidence from top candidates."""
        if not report.target_candidates:
            return 0.0
        strong = [t for t in report.target_candidates if t.tier == "strong"]
        pool = strong or report.target_candidates[:3]
        avg = sum(t.confidence for t in pool) / len(pool)
        return avg * 100
