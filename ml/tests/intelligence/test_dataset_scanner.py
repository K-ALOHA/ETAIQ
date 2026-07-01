"""Tests for dataset scanner."""

from __future__ import annotations

from ml.intelligence.dataset_scanner import (
    DatasetScanner,
    dataset_id_from_path,
    discover_csv_files,
)


def test_discover_csv_files_recursively(raw_data_dir) -> None:
    files = discover_csv_files(raw_data_dir)
    assert len(files) == 3


def test_dataset_id_from_nested_path(raw_data_dir) -> None:
    nested_file = raw_data_dir / "nested" / "riders.csv"
    dataset_id = dataset_id_from_path(nested_file, raw_data_dir)
    assert dataset_id == "nested_riders"


def test_scanner_loads_all_datasets(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    assert len(scanned) == 3
    assert len(frames) == 3
    assert all(item.row_count == 3 for item in scanned)
