"""Tests for the feature encoding framework."""

import csv
from pathlib import Path

from ml.features.config import FeatureEngineeringConfig
from ml.features.encoding import EncodingEngine
from ml.features.feature_pipeline import FeaturePipeline
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger


def test_encoding_plan_exports_and_assigns_strategies() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    training_df = pipeline.run()
    assert training_df is not None

    engine = EncodingEngine(config=config, logger=logger)
    plan = engine.prepare_encoding_plan(registry.registry)
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
        row["encoding_strategy"] in {"OneHot Encoding", "Ordinal Encoding"}
        for row in categorical_rows
    )
