"""Tests for quality score calculation."""

from __future__ import annotations

from ml.validation.models import ValidationResult
from ml.validation.quality_score import QualityScoreCalculator


def _result(name: str, score: float, dataset: str = "orders") -> ValidationResult:
    return ValidationResult(
        validator_name=name,
        dataset_name=dataset,
        passed=score >= 80,
        score=score,
        details={},
        duration_seconds=0.1,
    )


def test_quality_score_perfect_scores() -> None:
    results = [
        _result("schema", 100.0, "restaurants"),
        _result("nulls", 100.0, "restaurants"),
        _result("duplicates", 100.0, "restaurants"),
        _result("gps", 100.0, "restaurants"),
        _result("foreign_key", 100.0),
        _result("target", 100.0),
    ]
    overall, components = QualityScoreCalculator().calculate(results)
    assert overall == 100.0
    assert components["schema"] == 100.0


def test_quality_score_weighted_average() -> None:
    results = [
        _result("schema", 80.0, "restaurants"),
        _result("nulls", 100.0, "restaurants"),
        _result("duplicates", 100.0, "restaurants"),
        _result("gps", 100.0, "restaurants"),
        _result("foreign_key", 100.0),
        _result("target", 100.0),
    ]
    overall, _ = QualityScoreCalculator().calculate(results)
    assert 80.0 <= overall < 100.0


def test_quality_score_report_dict() -> None:
    payload = QualityScoreCalculator.to_report_dict(87.5, {"schema": 90.0})
    assert payload["overall_score"] == 87.5
    assert payload["scale"] == "0-100"
    assert "weights" in payload
