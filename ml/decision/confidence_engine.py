"""Confidence Engine to assign and calibrate confidence, priority, evidence, and rationale."""

from __future__ import annotations

from typing import Any

from ml.decision.logging_config import get_logger
from ml.decision.models import CleaningRecommendation, PriorityLevel

logger = get_logger(__name__)


class ConfidenceEngine:
    """Refines and assigns confidence scores, evidence, rationale, and priority."""

    def process(
        self,
        recommendations: list[CleaningRecommendation],
        reports: dict[str, Any],
    ) -> list[CleaningRecommendation]:
        """Calibrate confidence scores and priorities for all recommendations.

        Args:
            recommendations: List of raw cleaning recommendations.
            reports: Input reports context.

        Returns:
            list[CleaningRecommendation]: Refined cleaning recommendations.
        """
        logger.info("confidence_engine_process_start", count=len(recommendations))

        for rec in recommendations:
            self._calibrate(rec, reports)

        logger.info("confidence_engine_process_end")
        return recommendations

    def _calibrate(self, rec: CleaningRecommendation, reports: dict[str, Any]) -> None:
        """Calibrate a single recommendation's priority, confidence, and justification.

        For example, adjust priority based on whether the column is a primary target.
        """
        # Read target candidates to check if a column is a target candidate
        target_info = reports.get("target_candidates", {})
        strong_targets = {
            (t.get("dataset_id"), t.get("column"))
            for t in target_info.get("strong_targets", [])
        }

        # Check if the recommendation targets a strong prediction target
        is_target = (rec.dataset_id, rec.column) in strong_targets

        # If a target column has an issue (nulls, outliers), raise priority to CRITICAL
        if is_target and rec.priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM):
            rec.priority = PriorityLevel.CRITICAL
            rec.rationale = f"[Target Column] {rec.rationale}"
            if "Target column issue" not in rec.evidence:
                rec.evidence.append("Target column issue")

        # Basic validation to ensure fields are populated
        if not rec.evidence:
            rec.evidence = ["No issues or transformations flagged by previous engines."]
        if not rec.rationale:
            rec.rationale = "Default action determined by system heuristics."

        # Keep confidence within bounds [0.0, 1.0]
        rec.confidence = max(0.0, min(rec.confidence, 1.0))
