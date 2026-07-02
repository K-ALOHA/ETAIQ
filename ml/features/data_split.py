"""Reusable train/test split module for ETAIQ preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FeatureEngineeringConfig
from .feature_registry import FeatureRegistryManager
from .logging_config import FeatureEngineeringLogger


class DataSplitEngine:
    """Engine responsible for splitting the engineered dataset into training and testing sets."""

    TARGET_COLUMN = "actual_delivery_time_min"
    DATASET_FILENAME = "engineered_training_dataset.csv"
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    SHUFFLE = True

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
        registry_manager: FeatureRegistryManager | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.registry_manager = registry_manager or FeatureRegistryManager(config=self.config, logger=self.logger)

    def load_dataset(self) -> pd.DataFrame:
        """Load the fully engineered training dataset from disk."""
        dataset_path = Path(self.config.project_root) / "ml" / "data" / "features" / self.DATASET_FILENAME
        if not dataset_path.exists():
            raise FileNotFoundError(f"Engineered dataset not found: {dataset_path}")

        dataframe = pd.read_csv(dataset_path)
        self.logger.info(
            "Dataset loaded",
            path=str(dataset_path),
            rows=len(dataframe),
            columns=dataframe.shape[1],
        )
        return dataframe

    def split_dataset(
        self,
        engineered_dataset: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the engineered dataset into training and testing datasets."""
        if self.TARGET_COLUMN not in engineered_dataset.columns:
            raise ValueError(f"Target column not found: {self.TARGET_COLUMN}")

        registry = self.registry_manager.inspect_features(engineered_dataset)
        identifier_columns = [
            feature.name
            for feature in registry.features
            if feature.feature_type == "Identifier"
        ]
        self.logger.info("Identifiers removed", identifiers=identifier_columns)

        X = engineered_dataset.drop(columns=identifier_columns + [self.TARGET_COLUMN], errors="ignore")
        y = engineered_dataset[self.TARGET_COLUMN].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            shuffle=self.SHUFFLE,
        )

        self.logger.info(
            "Train/test split completed",
            total_rows=len(engineered_dataset),
            training_rows=len(X_train),
            testing_rows=len(X_test),
        )

        self._print_summary(
            total_rows=len(engineered_dataset),
            training_rows=len(X_train),
            testing_rows=len(X_test),
            training_features=X_train.shape[1],
            testing_features=X_test.shape[1],
            target_column=self.TARGET_COLUMN,
        )

        return X_train, X_test, y_train, y_test

    def _print_summary(
        self,
        total_rows: int,
        training_rows: int,
        testing_rows: int,
        training_features: int,
        testing_features: int,
        target_column: str,
    ) -> None:
        print("=" * 40)
        print("Train/Test Split Summary")
        print("=" * 40)
        print(f"Total Rows: {total_rows}")
        print(f"Training Rows: {training_rows}")
        print(f"Testing Rows: {testing_rows}")
        print(f"Training Features: {training_features}")
        print(f"Testing Features: {testing_features}")
        print(f"Target Column: {target_column}")
        print("=" * 40)
