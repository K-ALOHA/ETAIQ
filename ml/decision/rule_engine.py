"""Rule engine to scan input JSON reports and detect actionable data issues."""

from __future__ import annotations

from typing import Any

from ml.decision.logging_config import get_logger
from ml.decision.models import ActionableIssue, PriorityLevel

logger = get_logger(__name__)


class RuleEngine:
    """Identifies data quality, schema, leakage, and integrity issues from reports."""

    def __init__(self) -> None:
        pass

    def analyze(self, reports: dict[str, Any]) -> list[ActionableIssue]:
        """Run all detection rules across the input reports.

        Args:
            reports: Dictionary mapping report names (e.g., 'dataset_profile')
                     to loaded JSON dicts.

        Returns:
            list[ActionableIssue]: List of issues identified.
        """
        logger.info("rule_engine_analysis_start")
        issues: list[ActionableIssue] = []

        # 1. Analyze Validation Report
        val_rep = reports.get("validation_report", {})
        if val_rep:
            issues.extend(self._analyze_validation_report(val_rep))

        # 2. Analyze Dataset Profile for outliers and other stats
        profile_rep = reports.get("dataset_profile", {})
        if profile_rep:
            issues.extend(self._analyze_dataset_profile(profile_rep))

        # 3. Analyze Feature Candidates for PII and scaling/encoding recommendations
        feat_rep = reports.get("feature_candidates", {})
        if feat_rep:
            issues.extend(self._analyze_feature_candidates(feat_rep))

        # 4. Analyze Target Candidates and Leakage Report
        leak_rep = reports.get("leakage_report", {})
        if leak_rep:
            issues.extend(self._analyze_leakage_report(leak_rep))

        target_rep = reports.get("target_candidates", {})
        if target_rep:
            issues.extend(self._analyze_target_candidates(target_rep, profile_rep))

        logger.info("rule_engine_analysis_end", issues_found=len(issues))
        return issues

    def _analyze_validation_report(self, val_rep: dict[str, Any]) -> list[ActionableIssue]:
        """Extract issues from validation results."""
        issues: list[ActionableIssue] = []
        results = val_rep.get("results", [])

        for result in results:
            v_name = result.get("validator_name")
            ds_name = result.get("dataset_name")
            details = result.get("details", {})

            if v_name == "schema":
                missing = details.get("missing_columns", [])
                extra = details.get("extra_columns", [])
                if missing:
                    issues.append(
                        ActionableIssue(
                            issue_type="schema_missing_columns",
                            dataset_id=ds_name,
                            column=None,
                            description=f"Dataset {ds_name} is missing expected columns: {', '.join(missing)}",
                            severity=PriorityLevel.CRITICAL,
                            source_report="validation_report.json",
                            details={"missing_columns": missing},
                        )
                    )
                if extra:
                    issues.append(
                        ActionableIssue(
                            issue_type="schema_extra_columns",
                            dataset_id=ds_name,
                            column=None,
                            description=f"Dataset {ds_name} contains extra unexpected columns: {', '.join(extra)}",
                            severity=PriorityLevel.LOW,
                            source_report="validation_report.json",
                            details={"extra_columns": extra},
                        )
                    )

            elif v_name == "nulls":
                violations = details.get("non_nullable_violations", [])
                per_col = details.get("per_column", {})
                for col in violations:
                    col_info = per_col.get(col, {})
                    null_pct = col_info.get("percentage", 0.0)
                    issues.append(
                        ActionableIssue(
                            issue_type="null_violation",
                            dataset_id=ds_name,
                            column=col,
                            description=f"Non-nullable column {col} in {ds_name} contains null values ({null_pct}% nulls)",
                            severity=PriorityLevel.HIGH,
                            source_report="validation_report.json",
                            details={"null_percentage": null_pct},
                        )
                    )

            elif v_name == "duplicates":
                exact_dups = details.get("exact_duplicate_rows", 0)
                if exact_dups > 0:
                    issues.append(
                        ActionableIssue(
                            issue_type="duplicate_rows",
                            dataset_id=ds_name,
                            column=None,
                            description=f"Dataset {ds_name} contains {exact_dups} exact duplicate rows",
                            severity=PriorityLevel.HIGH,
                            source_report="validation_report.json",
                            details={"duplicate_row_count": exact_dups},
                        )
                    )

            elif v_name == "foreign_key":
                per_col = details.get("per_column", {})
                for col, info in per_col.items():
                    orphan_count = info.get("orphan_count", 0)
                    if orphan_count > 0:
                        ref_missing = info.get("reference_dataset_missing", False)
                        desc = (
                            f"Foreign key {col} in {ds_name} has {orphan_count} orphans "
                            f"because reference dataset is missing"
                            if ref_missing
                            else f"Foreign key {col} in {ds_name} has {orphan_count} orphan keys"
                        )
                        issues.append(
                            ActionableIssue(
                                issue_type="foreign_key_orphan",
                                dataset_id=ds_name,
                                column=col,
                                description=desc,
                                severity=PriorityLevel.HIGH if ref_missing else PriorityLevel.MEDIUM,
                                source_report="validation_report.json",
                                details={"orphan_count": orphan_count, "ref_missing": ref_missing},
                            )
                        )

            elif v_name == "gps":
                inv_counts = details.get("invalid_counts", {})
                for col, inv_count in inv_counts.items():
                    if inv_count > 0:
                        issues.append(
                            ActionableIssue(
                                issue_type="gps_invalid_bounds",
                                dataset_id=ds_name,
                                column=col,
                                description=f"GPS coordinate column {col} in {ds_name} has {inv_count} out-of-bound values",
                                severity=PriorityLevel.HIGH,
                                source_report="validation_report.json",
                                details={"invalid_count": inv_count},
                            )
                        )

        return issues

    def _analyze_dataset_profile(self, profile_rep: dict[str, Any]) -> list[ActionableIssue]:
        """Scan column statistics for outliers and anomalous values."""
        issues: list[ActionableIssue] = []
        datasets = profile_rep.get("datasets", [])

        for ds in datasets:
            ds_name = ds.get("dataset_id")
            columns = ds.get("columns", [])

            for col in columns:
                col_name = col.get("name")
                stats = col.get("statistics", {})
                null_pct = col.get("null_percentage", 0.0)

                # Check general high null count (not caught by schema validations)
                if null_pct > 5.0:
                    issues.append(
                        ActionableIssue(
                            issue_type="high_null_rate",
                            dataset_id=ds_name,
                            column=col_name,
                            description=f"Column {col_name} in {ds_name} has {null_pct}% null rate",
                            severity=PriorityLevel.MEDIUM if null_pct < 50.0 else PriorityLevel.HIGH,
                            source_report="dataset_profile.json",
                            details={"null_percentage": null_pct},
                        )
                    )

                # Outlier check based on mean/std dev
                if stats and "mean" in stats and "std" in stats:
                    mean = stats.get("mean")
                    std = stats.get("std")
                    c_min = stats.get("min")
                    c_max = stats.get("max")

                    if std is not None and mean is not None and std > 0:
                        # Check positive-only heuristics
                        if c_min is not None and c_min < 0:
                            # order_value, capacity, ratings, delivery times should never be negative
                            if any(w in col_name.lower() for w in ["value", "capacity", "rating", "time", "min", "eta"]):
                                issues.append(
                                    ActionableIssue(
                                        issue_type="negative_value_anomaly",
                                        dataset_id=ds_name,
                                        column=col_name,
                                        description=f"Column {col_name} in {ds_name} contains negative value ({c_min})",
                                        severity=PriorityLevel.HIGH,
                                        source_report="dataset_profile.json",
                                        details={"min": c_min},
                                    )
                                )

                        # Extreme value outliers
                        upper_bound = mean + 3 * std
                        lower_bound = mean - 3 * std

                        if (c_max is not None and c_max > upper_bound) or (c_min is not None and c_min < lower_bound):
                            issues.append(
                                ActionableIssue(
                                    issue_type="outlier_detected",
                                    dataset_id=ds_name,
                                    column=col_name,
                                    description=f"Column {col_name} in {ds_name} contains values outside 3-std deviation bounds (min={c_min}, max={c_max}, bounds=[{round(lower_bound, 2)}, {round(upper_bound, 2)}])",
                                    severity=PriorityLevel.MEDIUM,
                                    source_report="dataset_profile.json",
                                    details={
                                        "min": c_min,
                                        "max": c_max,
                                        "mean": mean,
                                        "std": std,
                                        "upper_bound": upper_bound,
                                        "lower_bound": lower_bound,
                                    },
                                )
                            )

        return issues

    def _analyze_feature_candidates(self, feat_rep: dict[str, Any]) -> list[ActionableIssue]:
        """Detect columns needing scaling, encoding, or removal (PII)."""
        issues: list[ActionableIssue] = []
        classifications = feat_rep.get("classifications", {})

        # PII
        for item in classifications.get("pii", []):
            issues.append(
                ActionableIssue(
                    issue_type="pii_column",
                    dataset_id=item.get("dataset_id"),
                    column=item.get("column"),
                    description=f"Column {item.get('column')} in {item.get('dataset_id')} contains PII data",
                    severity=PriorityLevel.HIGH,
                    source_report="feature_candidates.json",
                    details=item,
                )
            )

        # High Cardinality / Categorical needing encoding
        # (We only recommend actions for columns we want to KEEP for training but require transformations)
        # Note: rule engine gathers ALL raw findings/transformations.
        for item in classifications.get("recommended_feature", []):
            encoding = item.get("encoding")
            scaling = item.get("scaling")
            if encoding and encoding != "none":
                issues.append(
                    ActionableIssue(
                        issue_type="categorical_needs_encoding",
                        dataset_id=item.get("dataset_id"),
                        column=item.get("column"),
                        description=f"Categorical column {item.get('column')} in {item.get('dataset_id')} needs encoding ({encoding})",
                        severity=PriorityLevel.INFO,
                        source_report="feature_candidates.json",
                        details={"encoding": encoding},
                    )
                )
            if scaling and scaling != "none":
                issues.append(
                    ActionableIssue(
                        issue_type="numeric_needs_scaling",
                        dataset_id=item.get("dataset_id"),
                        column=item.get("column"),
                        description=f"Numeric column {item.get('column')} in {item.get('dataset_id')} needs scaling ({scaling})",
                        severity=PriorityLevel.INFO,
                        source_report="feature_candidates.json",
                        details={"scaling": scaling},
                    )
                )

        return issues

    def _analyze_leakage_report(self, leak_rep: dict[str, Any]) -> list[ActionableIssue]:
        """Detect columns with target leakage risks."""
        issues: list[ActionableIssue] = []
        findings = leak_rep.get("findings", [])

        for finding in findings:
            severity_str = finding.get("severity", "MEDIUM")
            priority = {
                "HIGH": PriorityLevel.HIGH,
                "MEDIUM": PriorityLevel.MEDIUM,
                "LOW": PriorityLevel.LOW,
            }.get(severity_str, PriorityLevel.MEDIUM)

            issues.append(
                ActionableIssue(
                    issue_type="target_leakage",
                    dataset_id=finding.get("dataset_id"),
                    column=finding.get("column"),
                    description=f"Column {finding.get('column')} in {finding.get('dataset_id')} leaks target {finding.get('related_target')} ({finding.get('rationale')})",
                    severity=priority,
                    source_report="leakage_report.json",
                    details=finding,
                )
            )

        return issues

    def _analyze_target_candidates(
        self, target_rep: dict[str, Any], profile_rep: dict[str, Any]
    ) -> list[ActionableIssue]:
        """Verify the health of the identified prediction targets."""
        issues: list[ActionableIssue] = []
        strong = target_rep.get("strong_targets", [])

        # Find target column null rate
        dataset_profiles = profile_rep.get("datasets", [])
        null_map = {}
        for ds in dataset_profiles:
            ds_id = ds.get("dataset_id")
            for col in ds.get("columns", []):
                null_map[(ds_id, col.get("name"))] = col.get("null_percentage", 0.0)

        for target in strong:
            ds_id = target.get("dataset_id")
            col_name = target.get("column")
            null_pct = null_map.get((ds_id, col_name), 0.0)

            if null_pct > 0.0:
                issues.append(
                    ActionableIssue(
                        issue_type="target_contains_nulls",
                        dataset_id=ds_id,
                        column=col_name,
                        description=f"Primary prediction target {col_name} in {ds_id} contains {null_pct}% nulls",
                        severity=PriorityLevel.CRITICAL,
                        source_report="target_candidates.json",
                        details={"null_percentage": null_pct},
                    )
                )

        return issues
