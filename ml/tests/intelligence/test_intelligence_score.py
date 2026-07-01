"""Tests for intelligence score calculator."""

from __future__ import annotations

from ml.intelligence.intelligence_score import IntelligenceScoreCalculator
from ml.intelligence.main import IntelligenceEngine


def test_intelligence_score_calculator(raw_data_dir, tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    report = IntelligenceEngine(raw_data_dir, reports_dir).run()
    scores = IntelligenceScoreCalculator().calculate(report)
    assert "overall_intelligence_score" in scores
    assert 0 <= scores["overall_intelligence_score"] <= 100
    assert scores["discovery_quality"] > 0


def test_intelligence_score_in_report(raw_data_dir, tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    report = IntelligenceEngine(raw_data_dir, reports_dir).run()
    assert report.intelligence_score
    assert report.intelligence_score["target_confidence"] > 0
