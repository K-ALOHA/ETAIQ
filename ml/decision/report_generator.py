"""Report generator to write cleaning plans and summaries to disk."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.decision.logging_config import get_logger
from ml.decision.models import (
    ApprovalManifestEntry,
    CleaningRecommendation,
    ExpectedImpacts,
)

logger = get_logger(__name__)


class DecisionReportGenerator:
    """Serializes and writes all decision reports and plans to the reports directory."""

    def __init__(self, reports_dir: Path) -> None:
        """Initialize the report generator.

        Args:
            reports_dir: Directory where reports will be saved.
        """
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        recommendations: list[CleaningRecommendation],
        manifest: list[ApprovalManifestEntry],
        impacts: ExpectedImpacts,
        total_issues: int,
    ) -> dict[str, Path]:
        """Write all decision-related reports to disk.

        Args:
            recommendations: Cleaning recommendations.
            manifest: Approval manifest entries.
            impacts: Expected quality impacts.
            total_issues: Total raw issues identified.

        Returns:
            dict[str, Path]: Map of report names to written file paths.
        """
        logger.info("report_generation_start", dir=str(self._reports_dir))

        paths = {
            "cleaning_plan_json": self._write_cleaning_plan_json(recommendations),
            "cleaning_plan_md": self._write_cleaning_plan_md(recommendations, manifest, impacts),
            "decision_summary_json": self._write_decision_summary_json(recommendations, manifest, total_issues),
            "estimated_quality_json": self._write_estimated_quality_json(impacts),
            "approval_manifest_json": self._write_approval_manifest_json(manifest),
        }

        logger.info("report_generation_end", files=len(paths))
        return paths

    def _write_json(self, filename: str, payload: Any) -> Path:
        """Write a dictionary or list to a JSON file."""
        path = self._reports_dir / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def _write_cleaning_plan_json(self, recommendations: list[CleaningRecommendation]) -> Path:
        """Write the cleaning_plan.json."""
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "recommendations": [
                {
                    "action": rec.action.value,
                    "dataset_id": rec.dataset_id,
                    "column": rec.column,
                    "confidence": rec.confidence,
                    "priority": rec.priority.value,
                    "rationale": rec.rationale,
                    "evidence": rec.evidence,
                    "details": rec.details,
                }
                for rec in recommendations
            ],
        }
        return self._write_json("cleaning_plan.json", payload)

    def _write_approval_manifest_json(self, manifest: list[ApprovalManifestEntry]) -> Path:
        """Write the approval_manifest.json."""
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "manifest": [
                {
                    "decision_id": entry.decision_id,
                    "dataset_id": entry.dataset_id,
                    "column": entry.column,
                    "action": entry.action.value,
                    "priority": entry.priority.value,
                    "confidence": entry.confidence,
                    "rationale": entry.rationale,
                    "status": entry.status,
                }
                for entry in manifest
            ],
        }
        return self._write_json("approval_manifest.json", payload)

    def _write_estimated_quality_json(self, impacts: ExpectedImpacts) -> Path:
        """Write the estimated_quality.json."""
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "metrics": {
                "data_quality": {
                    "baseline": impacts.data_quality.baseline,
                    "target": impacts.data_quality.target,
                    "delta": impacts.data_quality.delta,
                },
                "completeness": {
                    "baseline": impacts.completeness.baseline,
                    "target": impacts.completeness.target,
                    "delta": impacts.completeness.delta,
                },
                "consistency": {
                    "baseline": impacts.consistency.baseline,
                    "target": impacts.consistency.target,
                    "delta": impacts.consistency.delta,
                },
                "integrity": {
                    "baseline": impacts.integrity.baseline,
                    "target": impacts.integrity.target,
                    "delta": impacts.integrity.delta,
                },
                "model_reliability": {
                    "baseline": impacts.model_reliability.baseline,
                    "target": impacts.model_reliability.target,
                    "delta": impacts.model_reliability.delta,
                },
            },
        }
        return self._write_json("estimated_quality.json", payload)

    def _write_decision_summary_json(
        self,
        recommendations: list[CleaningRecommendation],
        manifest: list[ApprovalManifestEntry],
        total_issues: int,
    ) -> Path:
        """Write the decision_summary.json."""
        severities: dict[str, int] = {}
        actions: dict[str, int] = {}

        for rec in recommendations:
            severities[rec.priority.value] = severities.get(rec.priority.value, 0) + 1
            actions[rec.action.value] = actions.get(rec.action.value, 0) + 1

        avg_confidence = (
            sum(rec.confidence for rec in recommendations) / len(recommendations)
            if recommendations
            else 1.0
        )

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_issues_found": total_issues,
                "total_recommendations": len(recommendations),
                "overall_confidence": round(avg_confidence, 4),
            },
            "by_priority": severities,
            "by_action": actions,
        }
        return self._write_json("decision_summary.json", payload)

    def _write_cleaning_plan_md(
        self,
        recommendations: list[CleaningRecommendation],
        manifest: list[ApprovalManifestEntry],
        impacts: ExpectedImpacts,
    ) -> Path:
        """Write the cleaning_plan.md document."""
        lines = [
            "# ETAIQ Executive Cleaning Plan",
            "",
            f"**Generated:** {datetime.now(UTC).isoformat()}",
            "",
            "## Quality Improvement Forecast",
            "",
            "| Dimension | Baseline | Target | Expected Improvement |",
            "|-----------|---------:|-------:|---------------------:|",
            f"| **Overall Data Quality** | {impacts.data_quality.baseline:.2f}% | {impacts.data_quality.target:.2f}% | +{impacts.data_quality.delta:.2f}% |",
            f"| Completeness | {impacts.completeness.baseline:.2f}% | {impacts.completeness.target:.2f}% | +{impacts.completeness.delta:.2f}% |",
            f"| Consistency | {impacts.consistency.baseline:.2f}% | {impacts.consistency.target:.2f}% | +{impacts.consistency.delta:.2f}% |",
            f"| Integrity | {impacts.integrity.baseline:.2f}% | {impacts.integrity.target:.2f}% | +{impacts.integrity.delta:.2f}% |",
            f"| Model Reliability | {impacts.model_reliability.baseline:.2f}% | {impacts.model_reliability.target:.2f}% | +{impacts.model_reliability.delta:.2f}% |",
            "",
            "## Executive Summary",
            "",
            f"A total of {len(recommendations)} cleaning actions are recommended across datasets. "
            f"Implementing this cleaning plan will improve overall data quality from {impacts.data_quality.baseline}% "
            f"to {impacts.data_quality.target}%.",
            "",
            "## Actionable Manifest",
            "",
            "| ID | Dataset | Column | Action | Priority | Confidence | Rationale |",
            "|----|---------|--------|--------|----------|-----------:|-----------|",
        ]

        for entry in manifest:
            col_str = f"`{entry.column}`" if entry.column else "*Dataset Level*"
            lines.append(
                f"| {entry.decision_id} | `{entry.dataset_id}` | {col_str} | **{entry.action.value}** | `{entry.priority.value}` | {entry.confidence:.2%} | {entry.rationale} |"
            )

        lines.extend(
            [
                "",
                "## Step-by-Step Execution Sequence",
                "",
                "1. **Drop Structural Violations**: Drop target leakage and PII columns first.",
                "2. **De-duplicate Records**: Remove exact duplicate rows across all datasets.",
                "3. **Null Imputation**: Impute missing values for columns with moderate null counts.",
                "4. **Outlier Filtering**: Filter or flag records with anomalous numeric values.",
                "5. **Schema-Preserving Repair**: Fix invalid values, validate integrity, and preserve the original raw schema.",
                "",
            ]
        )

        content = "\n".join(lines)
        path = self._reports_dir / "cleaning_plan.md"
        path.write_text(content, encoding="utf-8")
        return path
