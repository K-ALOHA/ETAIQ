"""Tests for the ETAIQ feature scaling stage."""

import pandas as pd
from pathlib import Path

from ml.features.config import FeatureEngineeringConfig
from ml.features.encoding import EncodingEngine
from ml.features.logging_config import FeatureEngineeringLogger
from ml.features.scaling import ScalingEngine
from ml.features.data_split import DataSplitEngine
from ml.features.feature_registry import FeatureRegistryManager


def test_scaling_preserves_shape_and_exports_scaler() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)

    dataset_path = Path(config.project_root) / "ml" / "data" / "features" / "engineered_training_dataset.csv"
    engineered_dataset = pd.read_csv(dataset_path)
    engineered_dataset = engineered_dataset.sample(n=5000, random_state=42).reset_index(drop=True)

    split_engine = DataSplitEngine(config=config, logger=logger, registry_manager=registry)
    X_train, X_test, y_train, y_test = split_engine.split_dataset(engineered_dataset)

    encoding_engine = EncodingEngine(config=config, logger=logger)
    plan = encoding_engine.prepare_encoding_plan(registry.list_features())
    encoding_engine.export_encoding_plan(plan)
    encoding_engine.fit(X_train, plan=plan)
    encoded_X_train, encoded_X_test = encoding_engine.transform(X_train, X_test)

    scaling_engine = ScalingEngine(config=config, logger=logger)
    scaling_engine.fit(encoded_X_train)
    scaled_X_train, scaled_X_test = scaling_engine.transform(encoded_X_train, encoded_X_test)
    scaler_path = scaling_engine.export_scaler()

    assert len(scaled_X_train) == len(encoded_X_train)
    assert len(scaled_X_test) == len(encoded_X_test)
    assert scaled_X_train.shape[1] == encoded_X_train.shape[1]
    assert scaled_X_test.shape[1] == encoded_X_test.shape[1]
    assert list(scaled_X_train.columns) == list(encoded_X_train.columns)
    assert list(scaled_X_test.columns) == list(encoded_X_test.columns)
    assert Path(scaler_path).exists()

    numeric_columns = [
        col
        for col in encoded_X_train.columns
        if pd.api.types.is_numeric_dtype(encoded_X_train[col].dtype)
        and col not in scaling_engine.onehot_columns
        and col not in scaling_engine.boolean_columns
        and col not in scaling_engine.identifier_columns
        and col not in scaling_engine.ordinal_features
    ]

    if numeric_columns:
        means = scaled_X_train[numeric_columns].mean().abs()
        assert (means < 1e-6).all()

    onehot_columns = scaling_engine.onehot_columns
    for col in onehot_columns:
        assert scaled_X_train[col].isin([0, 1]).all()
        assert scaled_X_test[col].isin([0, 1]).all()

    for col in scaling_engine.boolean_columns:
        assert scaled_X_train[col].isin([True, False]).all()
        assert scaled_X_test[col].isin([True, False]).all()

    assert y_train.name == split_engine.TARGET_COLUMN
    assert y_test.name == split_engine.TARGET_COLUMN
