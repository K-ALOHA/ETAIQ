"""Approval manifest builder to structure cleaning actions for review."""

from __future__ import annotations

from ml.decision.logging_config import get_logger
from ml.decision.models import ApprovalManifestEntry, CleaningRecommendation, PriorityLevel

logger = get_logger(__name__)


class ApprovalManifestBuilder:
    """Structures actionable cleaning recommendations into an interactive approval manifest."""

    def build(self, recommendations: list[CleaningRecommendation]) -> list[ApprovalManifestEntry]:
        """Convert cleaning recommendations into sorted, uniquely identified approval entries.

        We sort entries:
        1. Priority (CRITICAL > HIGH > MEDIUM > LOW > INFO)
        2. Confidence (descending)
        3. Dataset ID
        4. Column Name (nulls last)
        """
        logger.info("approval_manifest_build_start", count=len(recommendations))

        # Sort priority mapping
        priority_weights = {
            PriorityLevel.CRITICAL: 5,
            PriorityLevel.HIGH: 4,
            PriorityLevel.MEDIUM: 3,
            PriorityLevel.LOW: 2,
            PriorityLevel.INFO: 1,
        }

        def sort_key(rec: CleaningRecommendation) -> tuple[int, float, str, str]:
            p_weight = priority_weights.get(rec.priority, 0)
            col_str = rec.column or ""
            return (-p_weight, -rec.confidence, rec.dataset_id, col_str)

        sorted_recs = sorted(recommendations, key=sort_key)

        manifest: list[ApprovalManifestEntry] = []
        for idx, rec in enumerate(sorted_recs, start=1):
            decision_id = f"DEC-{idx:03d}"
            manifest.append(
                ApprovalManifestEntry(
                    decision_id=decision_id,
                    dataset_id=rec.dataset_id,
                    column=rec.column,
                    action=rec.action,
                    priority=rec.priority,
                    confidence=rec.confidence,
                    rationale=rec.rationale,
                    status="PENDING_APPROVAL",
                )
            )

        logger.info("approval_manifest_build_end", entries=len(manifest))
        return manifest
