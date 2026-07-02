"""Tests for the feature encoding framework."""

import csv
from pathlib import Path

import pandas as pd

from ml.features.config import FeatureEngineeringConfig
from ml.features.encoding import EncodingEngine
from ml.features.feature_pipeline import FeaturePipeline
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger
from ml.features.models import FeatureMetadata, FeatureRegistry


def test_high_cardinality_categorical_is_skipped() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    engine = EncodingEngine(config=config, logger=logger)

    feature_registry = FeatureRegistry(
        features=[
            FeatureMetadata(name="high_cardinality_feature", feature_type="Categorical"),
            FeatureMetadata(name="low_cardinality_feature", feature_type="Categorical"),
        ]
    )
    X_train = pd.DataFrame(
        {
            "high_cardinality_feature": [f"value_{index % 150}" for index in range(150)],
            "low_cardinality_feature": ["A", "B"] * 75,
        }
    )

    plan = engine.prepare_encoding_plan(feature_registry, X_train=X_train)

    by_name = {entry.feature_name: entry.encoding_strategy for entry in plan.entries}
    assert by_name["high_cardinality_feature"] == "Skipped (High Cardinality)"
    assert by_name["low_cardinality_feature"] == "OneHot Encoding"


def test_encoding_plan_exports_and_assigns_strategies() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    training_df = pipeline.run()
    assert training_df is not None

    engine = EncodingEngine(config=config, logger=logger)
    plan = engine.prepare_encoding_plan(registry.registry, X_train=training_df)
    export_path = engine.export_encoding_plan(plan)

    assert Path(export_path).exists()

    with Path(export_path).open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    assert len(rows) == len(plan.entries)
    assert all(row["encoding_strategy"] for row in rows)

    identifier_rows = [row for row in rows if row["feature_type"] == "Identifier"]
    target_rows = [row for row in rows if row["feature_type"] == "Target"]
    ordinal_rows = [row for row in rows if row["feature_name"] in {"rider_experience_level", "restaurant_quality_tier"}]
    categorical_rows = [row for row in rows if row["feature_type"] == "Categorical"]

    assert identifier_rows
    assert target_rows
    assert all(row["encoding_strategy"] == "No Encoding" for row in identifier_rows)
    assert all(row["encoding_strategy"] == "No Encoding" for row in target_rows)
    assert all(row["encoding_strategy"] == "Ordinal Encoding" for row in ordinal_rows)
    assert all(
        row["encoding_strategy"] in {"OneHot Encoding", "Ordinal Encoding", "Skipped (High Cardinality)"}
        for row in categorical_rows
    )
