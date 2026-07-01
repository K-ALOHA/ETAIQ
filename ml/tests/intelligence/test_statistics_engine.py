"""Tests for statistics engine."""

from __future__ import annotations

from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.statistics_engine import StatisticsEngine


def test_statistics_engine_profiles_columns(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = StatisticsEngine().profile_all(frames, scanned)
    assert len(profiles) == 3
    orders = next(item for item in profiles if item.dataset_id == "orders")
    assert orders.row_count == 3
    assert orders.column_count >= 8
    assert orders.column_groups["numeric"]


def test_statistics_engine_detects_nulls(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    frames["orders"].loc[0, "customer_latitude"] = None
    profiles = StatisticsEngine().profile_all(frames, scanned)
    orders = next(item for item in profiles if item.dataset_id == "orders")
    lat = next(col for col in orders.columns if col.name == "customer_latitude")
    assert lat.null_count == 1
