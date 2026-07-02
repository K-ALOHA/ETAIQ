"""Tests for the ETAIQ train/test splitting module."""

import pandas as pd
from pathlib import Path

from ml.features.config import FeatureEngineeringConfig
from ml.features.data_split import DataSplitEngine
from ml.features.feature_pipeline import FeaturePipeline
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger


def test_data_split_preserves_total_row_count_and_removes_identifiers() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    engineered_dataset = pipeline.run()
    split_engine = DataSplitEngine(config=config, logger=logger, registry_manager=registry)

    X_train, X_test, y_train, y_test = split_engine.split_dataset(engineered_dataset)

    assert len(X_train) + len(X_test) == len(engineered_dataset)
    assert split_engine.TARGET_COLUMN not in X_train.columns
    assert split_engine.TARGET_COLUMN not in X_test.columns

    identifier_columns = [
        feature.name
        for feature in registry.list_features()
        if feature.feature_type == "Identifier"
    ]
    for identifier in identifier_columns:
        assert identifier not in X_train.columns
        assert identifier not in X_test.columns

    assert split_engine.TARGET_COLUMN in y_train.name
    assert split_engine.TARGET_COLUMN in y_test.name


def test_data_split_random_state_is_respected() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)

    engineered_dataset = pipeline.run()
    split_engine = DataSplitEngine(config=config, logger=logger, registry_manager=registry)

    first_run = split_engine.split_dataset(engineered_dataset)
    second_run = split_engine.split_dataset(engineered_dataset)

    assert first_run[0].reset_index(drop=True).equals(second_run[0].reset_index(drop=True))
    assert first_run[2].reset_index(drop=True).equals(second_run[2].reset_index(drop=True))
