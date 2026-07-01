"""Integration tests for intelligence CLI."""

from __future__ import annotations

from ml.intelligence.main import run_intelligence


def test_run_intelligence_end_to_end(raw_data_dir, tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    exit_code = run_intelligence(raw_data_dir, reports_dir)
    assert exit_code == 0
    assert (reports_dir / "dataset_profile.json").exists()
    assert (reports_dir / "schema_registry.json").exists()
    assert (reports_dir / "quality_score.json").exists() is False
