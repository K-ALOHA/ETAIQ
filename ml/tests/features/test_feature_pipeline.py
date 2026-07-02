"""Tests for the feature engineering data-loading and merge pipeline."""

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
