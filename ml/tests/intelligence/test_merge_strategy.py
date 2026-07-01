"""Tests for merge strategy builder."""

from __future__ import annotations

from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.merge_strategy import MergeStrategyBuilder
from ml.intelligence.relationship_detector import RelationshipDetector
from ml.intelligence.statistics_engine import StatisticsEngine


def test_merge_strategy_builder_creates_joins(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = StatisticsEngine().profile_all(frames, scanned)
    relationships, _ = RelationshipDetector().detect(frames, profiles)
    strategies = MergeStrategyBuilder().build(relationships)
    assert strategies
    assert all("join_type" in item for item in strategies)
