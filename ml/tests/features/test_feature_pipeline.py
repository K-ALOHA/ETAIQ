"""Tests for the feature engineering data-loading and merge pipeline."""

import csv
from pathlib import Path

import pandas as pd

from ml.features.config import FeatureEngineeringConfig
from ml.features.feature_engineering import FeatureEngineeringEngine
from ml.features.feature_pipeline import FeaturePipeline
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger


def test_configuration_loads() -> None:
    config = FeatureEngineeringConfig()
    assert config.random_seed == 42
    assert config.default_scaler_name == "standard_scaler"
    assert config.default_encoder_name == "one_hot_encoder"


def test_pipeline_initializes() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)
    assert pipeline is not None
    assert pipeline.state.status == "READY"


def test_registry_initializes() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    registry.initialize()
    assert registry.list_features() == []


def test_engine_initializes() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    engine = FeatureEngineeringEngine(config=config, logger=logger)
    assert engine is not None


def test_pipeline_loads_and_merges_data() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    training_df = pipeline.run()

    assert training_df is not None
    assert len(training_df) == len(pipeline.orders_df)
    assert training_df.shape[0] > 0
    assert not training_df.duplicated().any()
    assert "name" in training_df.columns
    assert "vehicle_type" in training_df.columns


def test_pipeline_creates_temporal_features() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    training_df = pipeline.run()

    assert "hour" in training_df.columns
    assert "weekday" in training_df.columns
    assert "meal_period" in training_df.columns
    assert "peak_hour" in training_df.columns
    assert "is_weekend" in training_df.columns
    assert len(training_df) == len(pipeline.orders_df)
    assert training_df["meal_period"].isin(["Breakfast", "Lunch", "Evening", "Night"]).all()
    assert training_df["hour"].between(0, 23).all()
    assert training_df["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).all()

    assert "distance_km" in training_df.columns
    assert "latitude_difference" in training_df.columns
    assert "longitude_difference" in training_df.columns
    assert "same_location" in training_df.columns
    assert (training_df["distance_km"] >= 0).all()

    assert "load_ratio" in training_df.columns
    assert "remaining_capacity" in training_df.columns
    assert "capacity_utilization_percent" in training_df.columns
    assert "orders_per_shift_hour" in training_df.columns
    assert "average_order_value_per_item" in training_df.columns
    assert "high_workload" in training_df.columns
    assert not training_df["load_ratio"].isin([float("inf"), float("-inf")]).any()
    assert not training_df["capacity_utilization_percent"].isin([float("inf"), float("-inf")]).any()
    assert not training_df["orders_per_shift_hour"].isin([float("inf"), float("-inf")]).any()

    assert "rider_experience_level" in training_df.columns
    assert "restaurant_quality_tier" in training_df.columns
    assert "high_value_order" in training_df.columns
    assert "large_order" in training_df.columns
    assert "premium_restaurant" in training_df.columns
    assert "busy_restaurant" in training_df.columns

    assert len(training_df) == len(pipeline.orders_df)

    assert training_df["rider_experience_level"].isin(["Beginner", "Intermediate", "Experienced", "Expert"]).all()
    assert training_df["restaurant_quality_tier"].isin(["Low", "Standard", "Good", "Premium"]).all()
    assert training_df["premium_restaurant"].dtype == bool
    assert training_df["busy_restaurant"].dtype == bool


def test_registry_classifies_string_columns_as_categorical() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    pipeline.load_data()
    pipeline.validate_data()
    pipeline.merge_data()
    pipeline.verify_merge()

    registry.inspect_features(pipeline.training_df)

    classification = {feature.name: feature.feature_type for feature in registry.list_features()}

    assert classification["vehicle_type"] == "Categorical"
    assert classification["cuisine"] == "Categorical"
    assert classification["promo_code_used"] == "Categorical"
    assert classification["order_status"] == "Categorical"


def test_pipeline_creates_feature_registry() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    if config.feature_registry_output_path.exists():
        config.feature_registry_output_path.unlink()

    training_df = pipeline.run()
    output_path = config.feature_registry_output_path

    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    assert len(rows) == len(registry.list_features())
    feature_names = {row["feature_name"] for row in rows}
    assert set(feature_names) == {feature.name for feature in registry.list_features()}
    assert any(row["feature_type"] == "Target" for row in rows)
    assert any(
        row["feature_type"] == "Identifier"
        for row in rows
        if row["feature_name"] in {"id", "restaurant_id", "rider_id"}
    )
    assert all(row["recommended_action"] for row in rows)


def test_pipeline_exports_engineered_dataset() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    export_path = Path(config.project_root) / "ml" / "data" / "features" / "engineered_training_dataset.csv"
    if export_path.exists():
        export_path.unlink()

    training_df = pipeline.run()

    assert export_path.exists()
    exported_df = pd.read_csv(export_path)

    assert len(exported_df) == len(training_df)
    assert exported_df.shape[1] == training_df.shape[1]
    assert exported_df.columns.tolist() == training_df.columns.tolist()
