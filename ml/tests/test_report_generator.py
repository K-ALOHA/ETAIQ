"""Tests for report generation."""

from __future__ import annotations

from pathlib import Path

from ml.validation.models import ValidationResult, ValidationSummary
from ml.validation.report_generator import ReportGenerator


def test_report_generator_writes_all_artifacts(tmp_path: Path) -> None:
    summary = ValidationSummary(
        results=[
            ValidationResult(
                validator_name="schema",
                dataset_name="orders",
                passed=True,
                score=100.0,
                details={"missing_columns": []},
                duration_seconds=0.01,
            )
        ],
        quality_score=100.0,
        component_scores={"schema": 100.0},
        data_dir="/data/raw",
        reports_dir=str(tmp_path),
    )

    paths = ReportGenerator(tmp_path).generate(summary)

    assert paths["validation_report_json"].exists()
    assert paths["validation_report_md"].exists()
    assert paths["quality_score_json"].exists()

    md_content = paths["validation_report_md"].read_text(encoding="utf-8")
    assert "ETAIQ Data Validation Report" in md_content
    assert "100.00" in md_content

    import json

    score_data = json.loads(paths["quality_score_json"].read_text(encoding="utf-8"))
    assert score_data["overall_score"] == 100.0
