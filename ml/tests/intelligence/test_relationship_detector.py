"""Tests for relationship detector."""

from __future__ import annotations

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.relationship_detector import RelationshipDetector
from ml.intelligence.statistics_engine import StatisticsEngine


def test_relationship_detector_finds_foreign_keys(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    relationships, fk_map = RelationshipDetector().detect(frames, profiles)
    assert relationships
    assert ("orders", "restaurant_id") in fk_map
    assert ("orders", "rider_id") in fk_map


def test_relationship_includes_confidence_and_reason(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = StatisticsEngine().profile_all(frames, scanned)
    relationships, _ = RelationshipDetector().detect(frames, profiles)
    assert relationships[0].join_confidence >= 0.55
    assert relationships[0].reason
    assert relationships[0].referential_integrity > 0


def test_relationship_registry_serialization(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = StatisticsEngine().profile_all(frames, scanned)
    relationships, _ = RelationshipDetector().detect(frames, profiles)
    registry = RelationshipDetector.to_registry(relationships)
    assert registry["relationship_count"] >= 1
    assert "reason" in registry["relationships"][0]


def test_relationship_detector_rejects_business_attributes(raw_data_dir) -> None:
    # Verify that business attributes like avg_rating, completed_orders, prep_capacity,
    # shift_hours, and order_size are rejected and never confirmed.
    scanned, frames = DatasetScanner(raw_data_dir).scan()

    # Inject business attributes into the frames to simulate the real dataset scenario
    frames["restaurants"]["avg_rating"] = [4.5, 3.8, 4.2]
    frames["restaurants"]["prep_capacity"] = [10, 15, 20]
    frames["nested_riders"]["shift_hours"] = [8.0, 6.0, 7.5]
    frames["nested_riders"]["completed_orders"] = [12, 10, 15]
    frames["orders"]["order_size"] = [2, 1, 3]

    # Re-profile to capture injected columns
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    relationships, fk_map = RelationshipDetector().detect(frames, profiles)

    # Ensure none of the rejected columns appear as source or target in any detected relationship
    rejected_attrs = {
        "avg_rating",
        "completed_orders",
        "prep_capacity",
        "shift_hours",
        "order_size",
    }
    for rel in relationships:
        assert rel.source_column not in rejected_attrs
        assert rel.target_column not in rejected_attrs
        assert rel.tier in ("Confirmed", "Likely")


def test_relationship_includes_detailed_metrics_and_sql(raw_data_dir) -> None:
    scanned, frames = DatasetScanner(raw_data_dir).scan()
    profiles = ColumnClassifier().classify_profiles(
        StatisticsEngine().profile_all(frames, scanned)
    )
    relationships, _ = RelationshipDetector().detect(frames, profiles)

    assert len(relationships) >= 1
    rel = relationships[0]

    # Verify report improvements are present
    assert rel.tier in ("Confirmed", "Likely")
    assert rel.confidence_breakdown
    assert "semantic_similarity" in rel.confidence_breakdown
    assert "referential_integrity" in rel.confidence_breakdown
    assert "cardinality" in rel.confidence_breakdown
    assert "datatype" in rel.confidence_breakdown
    assert "value_overlap" in rel.confidence_breakdown

    assert rel.business_justification
    assert rel.merge_risk
    assert rel.sql_join_example
    assert "JOIN" in rel.sql_join_example
    assert rel.source_column in rel.sql_join_example
