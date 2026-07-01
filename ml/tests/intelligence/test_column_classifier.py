"""Tests for column classifier."""

from __future__ import annotations

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.statistics_engine import StatisticsEngine


def test_column_classifier_assigns_roles(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = StatisticsEngine().profile_all(frames, scanned)
    profiles = ColumnClassifier().classify_profiles(profiles)
    orders = next(item for item in profiles if item.dataset_id == "orders")
    order_id = next(col for col in orders.columns if col.name == "order_id")
    assert "identifier" in order_id.roles


def test_column_classifier_detects_timestamp_columns(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    orders = next(item for item in profiles if item.dataset_id == "orders")
    ts_col = next(col for col in orders.columns if col.name == "order_timestamp")
    assert "timestamp" in ts_col.roles
