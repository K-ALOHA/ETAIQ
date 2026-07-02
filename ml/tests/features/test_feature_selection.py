"""Tests for the ETAIQ feature selection stage."""

import pandas as pd
from pathlib import Path

from ml.features.config import FeatureEngineeringConfig
from ml.features.data_split import DataSplitEngine
from ml.features.encoding import EncodingEngine
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger
from ml.features.scaling import ScalingEngine
from ml.features.selection import FeatureSelectionEngine


def test_feature_selection_exports_feature_lists_and_preserves_rows() -> None:
    config = FeatureEngineeringConfig()
    logger = FeatureEngineeringLogger(level="INFO")
    registry = FeatureRegistryManager(config=config, logger=logger)

    dataset_path = Path(config.project_root) / "ml" / "data" / "features" / "engineered_training_dataset.csv"
    engineered_dataset = pd.read_csv(dataset_path)
    engineered_dataset = engineered_dataset.sample(n=5000, random_state=42).reset_index(drop=True)

    split_engine = DataSplitEngine(config=config, logger=logger, registry_manager=registry)
    X_train, X_test, y_train, y_test = split_engine.split_dataset(engineered_dataset)

    encoding_engine = EncodingEngine(config=config, logger=logger)
    plan = encoding_engine.prepare_encoding_plan(registry.list_features(), X_train=X_train)
    encoding_engine.export_encoding_plan(plan)
    encoding_engine.fit(X_train, plan=plan)
    encoded_X_train, encoded_X_test = encoding_engine.transform(X_train, X_test)

    scaling_engine = ScalingEngine(config=config, logger=logger)
    scaling_engine.fit(encoded_X_train)
    scaled_X_train, scaled_X_test = scaling_engine.transform(encoded_X_train, encoded_X_test)

    selection_engine = FeatureSelectionEngine(config=config, logger=logger)
    selected_X_train, selected_X_test = selection_engine.select_features(scaled_X_train, scaled_X_test, y_train)

    feature_importance_path = Path(config.project_root) / "ml" / "data" / "features" / "feature_importance.csv"
    selected_features_path = Path(config.project_root) / "ml" / "data" / "features" / "selected_features.csv"

    assert len(selected_X_train) == len(scaled_X_train)
    assert len(selected_X_test) == len(scaled_X_test)
    assert list(selected_X_train.columns) == list(selected_X_test.columns)
    assert feature_importance_path.exists()
    assert selected_features_path.exists()
    assert selected_X_train.shape[1] > 0
    assert selected_X_test.shape[1] > 0
    assert len(selected_X_train.columns) == len(pd.read_csv(selected_features_path))
    assert selected_X_train.columns.nunique() == selected_X_train.shape[1]

    constant_columns = [column for column in selected_X_train.columns if selected_X_train[column].nunique(dropna=False) <= 1]
    assert constant_columns == []
    assert y_train.name == split_engine.TARGET_COLUMN
    assert y_test.name == split_engine.TARGET_COLUMN
