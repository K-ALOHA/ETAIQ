"""Validation report generation in JSON and Markdown formats."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ml.validation.models import ValidationSummary
from ml.validation.quality_score import QualityScoreCalculator


class ReportGenerator:
    """Writes validation reports and quality score artifacts to disk."""

    def __init__(self, reports_dir: Path) -> None:
        """Initialize the generator with an output directory.

        Args:
            reports_dir: Directory where report files will be written.
        """
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, summary: ValidationSummary) -> dict[str, Path]:
        """Generate all report artifacts.

        Args:
            summary: Completed validation summary.

        Returns:
            dict[str, Path]: Mapping of report type to written file path.
        """
        json_path = self._reports_dir / "validation_report.json"
        md_path = self._reports_dir / "validation_report.md"
        score_path = self._reports_dir / "quality_score.json"

        json_path.write_text(
            json.dumps(summary.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        md_path.write_text(self._build_markdown(summary), encoding="utf-8")
        score_path.write_text(
            json.dumps(
                QualityScoreCalculator.to_report_dict(
                    summary.quality_score,
                    summary.component_scores,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "validation_report_json": json_path,
            "validation_report_md": md_path,
            "quality_score_json": score_path,
        }

    def _build_markdown(self, summary: ValidationSummary) -> str:
        """Render a human-readable Markdown validation report.

        Args:
            summary: Completed validation summary.

        Returns:
            str: Markdown document content.
        """
        lines = [
            "# ETAIQ Data Validation Report",
            "",
            f"**Generated:** {datetime.now(UTC).isoformat()}",
            f"**Data directory:** `{summary.data_dir}`",
            f"**Overall quality score:** {summary.quality_score:.2f} / 100",
            "",
            "## Component Scores",
            "",
            "| Category | Score | Weight |",
            "|----------|------:|-------:|",
        ]

        weights = QualityScoreCalculator.to_report_dict(0, {})["weights"]
        for category, score in summary.component_scores.items():
            weight_pct = int(weights.get(category, 0) * 100)
            lines.append(f"| {category} | {score:.2f} | {weight_pct}% |")

        lines.extend(["", "## Validation Results", ""])

        for result in summary.results:
            status = "PASS" if result.passed else "FAIL"
            lines.extend(
                [
                    f"### {result.dataset_name} — {result.validator_name} ({status})",
                    "",
                    f"- **Score:** {result.score:.2f}",
                    f"- **Duration:** {result.duration_seconds:.4f}s",
                    "",
                ]
            )
            for key, value in result.details.items():
                lines.append(f"- **{key}:** `{value}`")
            lines.append("")

        passed = sum(1 for r in summary.results if r.passed)
        failed = len(summary.results) - passed
        lines.extend(
            [
                "## Summary",
                "",
                f"- Total checks: {len(summary.results)}",
                f"- Passed: {passed}",
                f"- Failed: {failed}",
                "",
            ]
        )
        return "\n".join(lines)
