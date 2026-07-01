"""Impact estimator to predict improvements across data quality dimensions."""

from __future__ import annotations

from typing import Any

from ml.decision.logging_config import get_logger
from ml.decision.models import ExpectedImpacts, ImpactMetric

logger = get_logger(__name__)


class ImpactEstimator:
    """Estimates expected improvements across various quality dimensions."""

    def estimate(
        self,
        reports: dict[str, Any],
        recommendations: list[Any],
    ) -> ExpectedImpacts:
        """Estimate baseline vs target improvements based on proposed recommendations.

        Args:
            reports: Input reports.
            recommendations: Cleaning recommendations list.

        Returns:
            ExpectedImpacts: Payload containing baseline, target, and delta values.
        """
        logger.info("impact_estimation_start")

        # 1. Base default metrics on Validation Report quality score if available
        val_rep = reports.get("validation_report", {})
        comp_scores = val_rep.get("component_scores", {})

        # Baseline metrics (0 to 100)
        base_nulls = comp_scores.get("nulls", 95.0)
        base_dups = comp_scores.get("duplicates", 95.0)
        base_schema = comp_scores.get("schema", 90.0)
        base_fk = comp_scores.get("foreign_key", 90.0)
        base_gps = comp_scores.get("gps", 100.0)

        # Count leakage and PII to calculate model reliability
        leak_rep = reports.get("leakage_report", {})
        leak_findings = len(leak_rep.get("findings", []))
        feat_rep = reports.get("feature_candidates", {})
        pii_columns = len(feat_rep.get("classifications", {}).get("pii", []))

        base_reliability = max(10.0, 100.0 - (leak_findings * 15.0 + pii_columns * 20.0))

        # Check what actions are recommended
        actions = {r.action for r in recommendations}

        # Impute / Drop high null rates -> expected completeness goes to 100.0
        target_completeness = 100.0 if "IMPUTE" in actions or "DROP" in actions else base_nulls

        # Remove duplicates / fix schemas -> expected consistency increases
        has_dup_action = "REMOVE_DUPLICATES" in actions
        has_schema_action = "FIX_DATATYPE" in actions or "DROP" in actions
        target_consistency = 100.0 if (has_dup_action or has_schema_action) else min(base_dups, base_schema)
        # Adjust target consistency to be high if we handle duplicates and schemas
        if has_dup_action and has_schema_action:
            target_consistency = 100.0
        elif has_dup_action:
            target_consistency = max(99.0, base_dups)
        else:
            target_consistency = (base_dups + base_schema) / 2.0

        # Enforce foreign key / GPS -> integrity increases
        target_integrity = 100.0 if "FLAG" in actions or "REMOVE_OUTLIERS" in actions else min(base_fk, base_gps)
        if "FLAG" in actions:
            target_integrity = max(98.0, base_fk)

        # Drop target leakage & PII -> model reliability goes to 100.0
        target_reliability = 100.0 if "DROP" in actions else base_reliability

        # Overall quality score
        baseline_quality = val_rep.get("quality_score", 70.0)
        target_quality = round(
            (target_completeness + target_consistency + target_integrity + target_reliability) / 4.0,
            2,
        )
        if target_quality > 98.0:
            target_quality = 98.5  # Real-world data is rarely 100% clean

        expected = ExpectedImpacts(
            data_quality=ImpactMetric(
                baseline=round(baseline_quality, 2),
                target=round(target_quality, 2),
                delta=round(target_quality - baseline_quality, 2),
            ),
            completeness=ImpactMetric(
                baseline=round(base_nulls, 2),
                target=round(target_completeness, 2),
                delta=round(target_completeness - base_nulls, 2),
            ),
            consistency=ImpactMetric(
                baseline=round(min(base_dups, base_schema), 2),
                target=round(target_consistency, 2),
                delta=round(target_consistency - min(base_dups, base_schema), 2),
            ),
            integrity=ImpactMetric(
                baseline=round(min(base_fk, base_gps), 2),
                target=round(target_integrity, 2),
                delta=round(target_integrity - min(base_fk, base_gps), 2),
            ),
            model_reliability=ImpactMetric(
                baseline=round(base_reliability, 2),
                target=round(target_reliability, 2),
                delta=round(target_reliability - base_reliability, 2),
            ),
        )

        logger.info(
            "impact_estimation_end",
            quality_delta=expected.data_quality.delta,
        )
        return expected
