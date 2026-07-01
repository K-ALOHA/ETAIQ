"""Tests for schema registry."""

from __future__ import annotations

from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.schema_registry import SchemaRegistry


def test_schema_registry_builds_dynamic_schema(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    registry = SchemaRegistry().build(frames, scanned)
    assert registry["dataset_count"] == 3
    assert "datasets" in registry
    for dataset in registry["datasets"].values():
        assert "columns" in dataset
        assert dataset["column_count"] >= 1
