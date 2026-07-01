"""Tests for intelligence report generator."""

from __future__ import annotations

from ml.intelligence.main import IntelligenceEngine


def test_report_generator_writes_all_artifacts(raw_data_dir, tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    report = IntelligenceEngine(raw_data_dir, reports_dir).run()

    from ml.intelligence.report_generator import IntelligenceReportGenerator

    paths = IntelligenceReportGenerator(reports_dir).generate(report)

    expected = [
        "dataset_profile.json",
        "dataset_profile.md",
        "schema_registry.json",
        "relationship_registry.json",
        "feature_candidates.json",
        "merge_strategy.json",
        "target_candidates.json",
        "leakage_report.json",
        "data_dictionary.md",
        "version_report.json",
        "intelligence_score.json",
        "relationship_graph.json",
    ]
    for filename in expected:
        assert (reports_dir / filename).exists(), f"missing {filename}"
    assert len(paths) == len(expected)


def test_target_report_has_tier_groups(raw_data_dir, tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    report = IntelligenceEngine(raw_data_dir, reports_dir).run()
    from ml.intelligence.report_generator import IntelligenceReportGenerator

    IntelligenceReportGenerator(reports_dir).generate(report)
    import json

    payload = json.loads(
        (reports_dir / "target_candidates.json").read_text(encoding="utf-8")
    )
    assert "strong_targets" in payload
    assert "summary" in payload
