"""Tests for target detector."""

from __future__ import annotations

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.statistics_engine import StatisticsEngine
from ml.intelligence.target_detector import TargetDetector


def test_target_detector_ranks_delivery_time_strong(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    assert targets
    strong = [t for t in targets if t.tier == "strong"]
    assert strong
    assert strong[0].column == "delivery_time_minutes"
    assert strong[0].confidence >= 0.85
    assert strong[0].evidence


def test_target_detector_limits_candidate_count(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    assert len(targets) < 10
    assert "order_id" not in {t.column for t in targets}


def test_target_detector_excludes_coordinates(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    target_columns = {item.column for item in targets}
    assert "latitude" not in target_columns
    assert "longitude" not in target_columns
