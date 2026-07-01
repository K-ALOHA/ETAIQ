"""Tests for feature recommender."""

from __future__ import annotations

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.feature_recommender import FeatureRecommender
from ml.intelligence.statistics_engine import StatisticsEngine


def test_feature_recommender_generates_candidates(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    candidates = FeatureRecommender().recommend(profiles)
    assert len(candidates) > 0
    assert all(0.0 <= item.confidence <= 1.0 for item in candidates)
    assert all(item.classification for item in candidates)
    assert all(item.reason for item in candidates)


def test_feature_recommender_classifies_identifiers(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    candidates = FeatureRecommender().recommend(profiles)
    order_id = next(item for item in candidates if item.column == "order_id")
    assert order_id.classification == "identifier"
    assert order_id.recommendation == "ignore"


def test_feature_recommender_ranks_by_confidence(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    candidates = FeatureRecommender().recommend(profiles)
    confidences = [c.confidence for c in candidates]
    assert confidences == sorted(confidences, reverse=True)
