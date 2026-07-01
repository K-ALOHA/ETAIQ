"""Tests for version tracker."""

from __future__ import annotations

import json

from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.schema_registry import SchemaRegistry
from ml.intelligence.version_tracker import VersionTracker


def test_version_tracker_initial_run(tmp_path, raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    registry = SchemaRegistry().build(frames, scanned)
    report = VersionTracker().compare(registry, tmp_path)
    assert report["status"] == "initial_version"


def test_version_tracker_detects_column_addition(tmp_path, raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    registry = SchemaRegistry().build(frames, scanned)
    (tmp_path / "schema_registry.json").write_text(
        json.dumps(registry),
        encoding="utf-8",
    )

    frames["orders"]["new_metric"] = [1, 2, 3]
    updated = SchemaRegistry().build(frames, scanned)
    report = VersionTracker().compare(updated, tmp_path)
    assert report["status"] == "compared"
    changes = report["changes"]["dataset_changes"]["orders"]
    assert "new_metric" in changes["added_columns"]
