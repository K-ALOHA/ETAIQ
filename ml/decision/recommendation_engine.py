"""Recommendation engine to generate deterministic cleaning actions from issues."""

from __future__ import annotations

from typing import Any

from ml.decision.logging_config import get_logger
from ml.decision.models import ActionableIssue, ActionVerb, CleaningRecommendation, PriorityLevel

logger = get_logger(__name__)


class RecommendationEngine:
    """Generates deterministic cleaning actions for all columns and tables."""

    def __init__(self, null_drop_threshold: float = 0.80) -> None:
        """Initialize the recommendation engine.

        Args:
            null_drop_threshold: Fraction of nulls above which columns are dropped.
        """
        self._null_drop_threshold = null_drop_threshold

    def generate(self, issues: list[ActionableIssue], reports: dict[str, Any]) -> list[CleaningRecommendation]:
        """Map actionable issues to cleaning recommendations.

        Also generates default KEEP/LEAVE_UNCHANGED actions for all columns
        that have no issues.

        Args:
            issues: Detected issues list.
            reports: Input reports context.

        Returns:
            list[CleaningRecommendation]: Generated recommendations.
        """
        logger.info("recommendation_engine_generate_start", issues=len(issues))
        recommendations: list[CleaningRecommendation] = []
        handled_columns: set[tuple[str, str | None]] = set()

        for issue in issues:
            rec = self._resolve_issue(issue)
            if rec:
                recommendations.append(rec)
                handled_columns.add((issue.dataset_id, issue.column))

        # Generate default actions for columns/datasets without issues
        recommendations.extend(self._generate_defaults(handled_columns, reports))

        logger.info("recommendation_engine_generate_end", recommendations=len(recommendations))
        return recommendations

    def _resolve_issue(self, issue: ActionableIssue) -> CleaningRecommendation | None:
        """Resolve a single issue into a recommendation."""
        itype = issue.issue_type
        ds_id = issue.dataset_id
        col = issue.column
        details = issue.details

        if itype == "schema_missing_columns":
            return None

        if itype == "schema_extra_columns":
            return None

        if itype == "null_violation":
            null_pct = details.get("null_percentage", 0.0)
            if null_pct > self._null_drop_threshold * 100:
                action = ActionVerb.DROP
                rationale = f"Null violation in non-nullable column exceeds drop threshold ({null_pct}% nulls). Recommend dropping column."
            else:
                action = ActionVerb.IMPUTE
                rationale = f"Null violation in non-nullable column ({null_pct}% nulls). Recommend imputation."
            return CleaningRecommendation(
                action=action,
                dataset_id=ds_id,
                column=col,
                confidence=0.95,
                priority=issue.severity,
                rationale=rationale,
                evidence=[f"Null percentage: {null_pct}%"],
                details=details,
            )

        if itype == "duplicate_rows":
            dup_count = details.get("duplicate_row_count", 0)
            return CleaningRecommendation(
                action=ActionVerb.REMOVE_DUPLICATES,
                dataset_id=ds_id,
                column=None,
                confidence=1.0,
                priority=issue.severity,
                rationale=f"Dataset contains {dup_count} exact duplicate rows. Remove duplicates to ensure consistency.",
                evidence=[f"Duplicate row count: {dup_count}"],
                details=details,
            )

        if itype == "foreign_key_orphan":
            orphan_count = details.get("orphan_count", 0)
            return CleaningRecommendation(
                action=ActionVerb.REPAIR_FOREIGN_KEY,
                dataset_id=ds_id,
                column=col,
                confidence=0.90,
                priority=issue.severity,
                rationale=f"Foreign key contains {orphan_count} orphans. Repair invalid references by removing orphan records.",
                evidence=[f"Orphan count: {orphan_count}"],
                details=details,
            )

        if itype == "gps_invalid_bounds":
            inv_count = details.get("invalid_count", 0)
            return CleaningRecommendation(
                action=ActionVerb.FIX_GPS,
                dataset_id=ds_id,
                column=col,
                confidence=0.95,
                priority=issue.severity,
                rationale=f"GPS coordinates contain {inv_count} invalid bound violations. Repair out-of-bounds values.",
                evidence=[f"Invalid bounds count: {inv_count}"],
                details=details,
            )

        if itype == "high_null_rate":
            null_pct = details.get("null_percentage", 0.0)
            if null_pct > self._null_drop_threshold * 100:
                action = ActionVerb.DROP
                rationale = f"High null rate ({null_pct}%) exceeds the drop threshold of {int(self._null_drop_threshold * 100)}%. Exclude column."
            else:
                action = ActionVerb.IMPUTE
                rationale = f"Moderate null rate ({null_pct}%) is safe for imputation."
            return CleaningRecommendation(
                action=action,
                dataset_id=ds_id,
                column=col,
                confidence=0.85,
                priority=issue.severity,
                rationale=rationale,
                evidence=[f"Null percentage: {null_pct}%"],
                details=details,
            )

        if itype == "negative_value_anomaly":
            c_min = details.get("min")
            return CleaningRecommendation(
                action=ActionVerb.REMOVE_OUTLIERS,
                dataset_id=ds_id,
                column=col,
                confidence=0.95,
                priority=issue.severity,
                rationale=f"Negative value ({c_min}) in positive-only column. Exclude negative outlier rows.",
                evidence=[f"Minimum value: {c_min}"],
                details=details,
            )

        if itype == "outlier_detected":
            return CleaningRecommendation(
                action=ActionVerb.REMOVE_OUTLIERS,
                dataset_id=ds_id,
                column=col,
                confidence=0.80,
                priority=issue.severity,
                rationale="Values exceed 3-std deviation boundary limits. Filter or cap outlier records.",
                evidence=[f"Min: {details.get('min')}, Max: {details.get('max')}"],
                details=details,
            )

        if itype == "pii_column":
            return CleaningRecommendation(
                action=ActionVerb.DROP,
                dataset_id=ds_id,
                column=col,
                confidence=1.0,
                priority=issue.severity,
                rationale="Personally Identifiable Information detected. Drop column to preserve privacy.",
                evidence=["Flagged by intelligence classifiers."],
                details=details,
            )

        if itype == "target_leakage":
            rationale = f"Target leakage risk: {details.get('rationale')}. Drop column to prevent target leakage."
            return CleaningRecommendation(
                action=ActionVerb.DROP,
                dataset_id=ds_id,
                column=col,
                confidence=0.98,
                priority=issue.severity,
                rationale=rationale,
                evidence=[f"Correlation score: {details.get('confidence')}"],
                details=details,
            )

        if itype == "target_contains_nulls":
            null_pct = details.get("null_percentage", 0.0)
            return CleaningRecommendation(
                action=ActionVerb.DROP,  # Drop rows having null targets
                dataset_id=ds_id,
                column=col,
                confidence=0.99,
                priority=issue.severity,
                rationale=f"Prediction target contains {null_pct}% nulls. Drop rows containing null targets since they cannot be imputed.",
                evidence=[f"Target null percentage: {null_pct}%"],
                details=details,
            )

        if itype == "categorical_needs_encoding":
            return None

        if itype == "numeric_needs_scaling":
            return None

        return None

    def _generate_defaults(self, handled: set[tuple[str, str | None]], reports: dict[str, Any]) -> list[CleaningRecommendation]:
        """Generate LEAVE_UNCHANGED/KEEP for all other columns."""
        defaults: list[CleaningRecommendation] = []
        schema_reg = reports.get("schema_registry", {})
        datasets = schema_reg.get("datasets", {})

        for ds_id, ds_info in datasets.items():
            # If the entire dataset has no actions generated so far, check if we need REMOVE_DUPLICATES
            # (If duplicate check passed, we still might want KEEP for the dataset grain)
            if (ds_id, None) not in handled:
                defaults.append(
                    CleaningRecommendation(
                        action=ActionVerb.LEAVE_UNCHANGED,
                        dataset_id=ds_id,
                        column=None,
                        confidence=1.0,
                        priority=PriorityLevel.INFO,
                        rationale="No dataset-level issues detected.",
                        evidence=["Passed duplicate check."],
                    )
                )

            columns = ds_info.get("columns", {})
            for col_name in columns.keys():
                if (ds_id, col_name) not in handled:
                    defaults.append(
                        CleaningRecommendation(
                            action=ActionVerb.KEEP,
                            dataset_id=ds_id,
                            column=col_name,
                            confidence=1.0,
                            priority=PriorityLevel.INFO,
                            rationale="No issues or transformations required for this feature. Keep as-is.",
                            evidence=["No anomalies detected."],
                        )
                    )

        return defaults
