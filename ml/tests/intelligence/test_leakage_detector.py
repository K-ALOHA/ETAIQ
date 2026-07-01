"""Tests for leakage detector."""

from __future__ import annotations

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.leakage_detector import LeakageDetector
from ml.intelligence.statistics_engine import StatisticsEngine
from ml.intelligence.target_detector import TargetDetector


def test_leakage_detector_finds_correlated_column(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    findings = LeakageDetector().detect(frames, profiles, targets)
    leaked_columns = {item.column for item in findings}
    assert "delivery_time_estimate" in leaked_columns


def test_leakage_detector_includes_recommendation(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    findings = LeakageDetector().detect(frames, profiles, targets)
    assert findings
    assert all(item.recommendation for item in findings)
    assert all(item.confidence > 0 for item in findings)


def test_leakage_detector_annotates_profiles(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    targets = TargetDetector().detect(profiles)
    findings = LeakageDetector().detect(frames, profiles, targets)
    LeakageDetector.annotate_profiles(profiles, findings)
    orders = next(item for item in profiles if item.dataset_id == "orders")
    estimate = next(
        col for col in orders.columns if col.name == "delivery_time_estimate"
    )
    assert "potential_leakage" in estimate.roles
