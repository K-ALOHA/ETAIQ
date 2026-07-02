"""Data-loading and merge pipeline for the feature engineering module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FeatureEngineeringConfig
from .feature_engineering import FeatureEngineeringEngine
from .feature_registry import FeatureRegistryManager
from .logging_config import FeatureEngineeringLogger
from .models import PipelineState


class FeaturePipeline:
    """Load, validate, merge, and verify processed ETAIQ datasets."""

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
        registry: FeatureRegistryManager | None = None,
        engine: FeatureEngineeringEngine | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.registry = registry or FeatureRegistryManager(config=self.config, logger=self.logger)
        self.engine = engine or FeatureEngineeringEngine(config=self.config, logger=self.logger, registry=self.registry)
        self.state = PipelineState()
        self.restaurants_df: pd.DataFrame | None = None
        self.riders_df: pd.DataFrame | None = None
        self.orders_df: pd.DataFrame | None = None
        self.training_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Load the processed datasets from the configured processed-data directory."""
        data_dir = Path(self.config.processed_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

        files = {
            "restaurants": data_dir / "restaurants.csv",
            "riders": data_dir / "riders.csv",
            "orders": data_dir / "orders.csv",
        }

        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Required data file not found: {path}")

        self.restaurants_df = pd.read_csv(files["restaurants"])
        self.riders_df = pd.read_csv(files["riders"])
        self.orders_df = pd.read_csv(files["orders"])

        required_columns = {
            "restaurants": ["id"],
            "riders": ["id"],
            "orders": ["id", "restaurant_id", "rider_id"],
        }

        for dataset_name, expected_columns in required_columns.items():
            df = getattr(self, f"{dataset_name}_df")
            if df is None:
                raise ValueError(f"Dataset {dataset_name} was not loaded.")
            if df.empty:
                raise ValueError(f"Dataset {dataset_name} is empty.")
            missing_columns = [column for column in expected_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f"Dataset {dataset_name} is missing required columns: {missing_columns}")

        self.logger.info("Datasets loaded successfully")

    def validate_data(self) -> None:
        """Validate the loaded datasets before performing the merge."""
        if self.restaurants_df is None or self.riders_df is None or self.orders_df is None:
            raise ValueError("Data not loaded. Call load_data() before validation.")

        restaurants = self.restaurants_df
        riders = self.riders_df
        orders = self.orders_df

        if restaurants["id"].duplicated().any():
            raise ValueError("Duplicate restaurant primary keys detected.")
        if riders["id"].duplicated().any():
            raise ValueError("Duplicate rider primary keys detected.")
        if orders["id"].duplicated().any():
            raise ValueError("Duplicate order primary keys detected.")

        missing_restaurant_keys = set(orders["restaurant_id"]) - set(restaurants["id"])
        missing_rider_keys = set(orders["rider_id"]) - set(riders["id"])

        if missing_restaurant_keys:
            raise ValueError(f"Orders reference missing restaurant ids: {sorted(list(missing_restaurant_keys))[:10]}")
        if missing_rider_keys:
            raise ValueError(f"Orders reference missing rider ids: {sorted(list(missing_rider_keys))[:10]}")

        if orders["restaurant_id"].dtype != restaurants["id"].dtype:
            raise ValueError(
                "Restaurant join key types do not match: "
                f"orders.restaurant_id={orders['restaurant_id'].dtype}, restaurants.id={restaurants['id'].dtype}"
            )
        if orders["rider_id"].dtype != riders["id"].dtype:
            raise ValueError(
                "Rider join key types do not match: "
                f"orders.rider_id={orders['rider_id'].dtype}, riders.id={riders['id'].dtype}"
            )

        self.logger.info("Validation completed successfully")

    def merge_data(self) -> None:
        """Merge restaurants and riders into the orders dataframe using left joins."""
        if self.restaurants_df is None or self.riders_df is None or self.orders_df is None:
            raise ValueError("Data not loaded. Call load_data() before merge_data().")

        self.training_df = self.orders_df.merge(
            self.restaurants_df,
            left_on="restaurant_id",
            right_on="id",
            how="left",
            suffixes=("", "_restaurant"),
        )
        self.training_df = self.training_df.merge(
            self.riders_df,
            left_on="rider_id",
            right_on="id",
            how="left",
            suffixes=("", "_rider"),
        )
        self.logger.info("Merge completed successfully")

    def verify_merge(self) -> None:
        """Verify the merged dataframe integrity and print a merge summary."""
        if self.training_df is None:
            raise ValueError("Training dataframe not created. Call merge_data() first.")
        if self.orders_df is None:
            raise ValueError("Orders dataframe not loaded.")

        if len(self.training_df) != len(self.orders_df):
            raise ValueError("Row count changed after merge.")
        if self.training_df.duplicated().any():
            raise ValueError("Duplicate rows introduced during merge.")

        restaurant_lookup_successful = self.training_df["restaurant_id"].isin(self.restaurants_df["id"]).all()
        rider_lookup_successful = self.training_df["rider_id"].isin(self.riders_df["id"]).all()
        if not restaurant_lookup_successful:
            raise ValueError("Restaurant join keys are not preserved for every order.")
        if not rider_lookup_successful:
            raise ValueError("Rider join keys are not preserved for every order.")

        unexpected_nulls = self.training_df[["restaurant_id", "rider_id"]].isna().any().any()
        if unexpected_nulls:
            raise ValueError("Unexpected null values were introduced in merge keys.")

        print("=" * 40)
        print("Merge Summary")
        print("=" * 40)
        print(f"Rows: {len(self.training_df)}")
        print(f"Columns: {self.training_df.shape[1]}")
        print(f"Duplicate Rows: {int(self.training_df.duplicated().sum())}")
        print(f"Restaurant Join Successful: {restaurant_lookup_successful}")
        print(f"Rider Join Successful: {rider_lookup_successful}")
        print("=" * 40)
        self.state.status = "READY"
        self.state.step = "merged"
        self.state.message = "Training dataframe created successfully."
        self.logger.info("Merge verification completed successfully")

    def engineer_features(self) -> None:
        """Placeholder for feature engineering execution."""
        self.logger.info("Placeholder: engineer_features")

    def encode_features(self) -> None:
        """Placeholder for encoding step."""
        self.logger.info("Placeholder: encode_features")

    def scale_features(self) -> None:
        """Placeholder for scaling step."""
        self.logger.info("Placeholder: scale_features")

    def select_features(self) -> None:
        """Placeholder for feature selection."""
        self.logger.info("Placeholder: select_features")

    def export(self) -> None:
        """Placeholder for exporting engineered features."""
        self.logger.info("Placeholder: export")

    def export_engineered_dataset(self) -> None:
        """Export the fully engineered training dataset to a CSV file."""
        if self.training_df is None:
            raise ValueError("Training dataframe is not available for export.")

        export_path = Path(self.config.project_root) / "ml" / "data" / "features" / "engineered_training_dataset.csv"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        self.training_df.to_csv(export_path, index=False)

        rows, columns = self.training_df.shape
        self.logger.info(
            "Engineered dataset exported",
            rows=rows,
            columns=columns,
            export_path=str(export_path),
        )

        print("=" * 40)
        print("Export Summary")
        print("=" * 40)
        print("Engineered Dataset Saved")
        print(f"Rows: {rows}")
        print(f"Columns: {columns}")
        print(f"Export Path: {export_path}")
        print("=" * 40)

    def run(self) -> pd.DataFrame:
        """Run the pipeline steps and return the merged training dataframe."""
        self.load_data()
        self.validate_data()
        self.merge_data()
        self.verify_merge()
        self.registry.inspect_features(self.training_df)
        self.registry.export_registry()
        self.training_df = self.engine.create_temporal_features(self.training_df)
        self.training_df = self.engine.create_geographical_features(self.training_df)
        self.training_df = self.engine.create_operational_features(self.training_df)
        self.training_df = self.engine.create_business_features(self.training_df)
        self.export_engineered_dataset()
        if self.training_df is None:
            raise ValueError("Training dataframe could not be created.")
        return self.training_df
