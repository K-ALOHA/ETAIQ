"""Tests for the ETAIQ encoding execution stage."""

import pandas as pd
from pathlib import Path

from ml.features.config import FeatureEngineeringConfig
from ml.features.data_split import DataSplitEngine
from ml.features.encoding import EncodingEngine
from ml.features.feature_registry import FeatureRegistryManager
from ml.features.logging_config import FeatureEngineeringLogger


def test_encoding_execution_fits_transforms_and_persists_encoders() -> None:
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
    onehot_path, ordinal_path = encoding_engine.export_encoders()

    assert len(encoded_X_train) == len(X_train)
    assert len(encoded_X_test) == len(X_test)
    assert list(encoded_X_train.columns) == list(encoded_X_test.columns)
    assert onehot_path.endswith("onehot_encoder.pkl")
    assert ordinal_path.endswith("ordinal_encoder.pkl")
    assert Path(onehot_path).exists()
    assert Path(ordinal_path).exists()
    assert y_train.name == split_engine.TARGET_COLUMN
    assert y_test.name == split_engine.TARGET_COLUMN

    assert any("_" in col and col.split("_")[0] in encoding_engine.onehot_features for col in encoded_X_train.columns)
    assert all(
        encoded_X_train[column].dtype.kind in {"i", "u", "f"}
        for column in encoding_engine.ordinal_features
    )

    if encoding_engine.onehot_features:
        X_test_unknown = X_test.copy()
        onehot_feature = encoding_engine.onehot_features[0]
        X_test_unknown.iloc[0, X_test_unknown.columns.get_loc(onehot_feature)] = "__unknown_category__"
        _, encoded_unknown = encoding_engine.transform(X_train, X_test_unknown)
        onehot_columns = [col for col in encoded_unknown.columns if col.startswith(f"{onehot_feature}_")]
        assert encoded_unknown.loc[X_test_unknown.index[0], onehot_columns].sum() == 0

    if encoding_engine.ordinal_features:
        X_test_unknown = X_test.copy()
        ordinal_feature = encoding_engine.ordinal_features[0]
        X_test_unknown.iloc[0, X_test_unknown.columns.get_loc(ordinal_feature)] = "__unknown_category__"
        _, encoded_unknown = encoding_engine.transform(X_train, X_test_unknown)
        assert encoded_unknown.loc[X_test_unknown.index[0], ordinal_feature] == -1
