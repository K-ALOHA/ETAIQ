"""Encoding architecture for ETAIQ feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import FeatureEngineeringConfig
from .encoding_registry import EncodingPlan, EncodingPlanEntry, EncodingRegistry
from .logging_config import FeatureEngineeringLogger
from .models import FeatureRegistry


class EncodingEngine:
    """Prepare an encoding plan from the feature registry and execute encoding."""

    ENCODER_DIR = Path("ml") / "models" / "preprocessing"
    ONEHOT_ENCODER_FILE = "onehot_encoder.pkl"
    ORDINAL_ENCODER_FILE = "ordinal_encoder.pkl"
    ENCODING_PLAN_FILE = Path("ml") / "data" / "features" / "encoding_plan.csv"

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.registry = EncodingRegistry(config=self.config, logger=self.logger)
        self.onehot_encoder: OneHotEncoder | None = None
        self.ordinal_encoder: OrdinalEncoder | None = None
        self.onehot_features: list[str] = []
        self.ordinal_features: list[str] = []
        self.non_encoded_features: list[str] = []

    def prepare_encoding_plan(self, feature_registry: FeatureRegistry) -> EncodingPlan:
        """Inspect the feature registry and generate an encoding plan."""
        plan = self.registry.create_plan(feature_registry)
        self.logger.info("Encoding plan generated", plan_size=len(plan))
        return plan

    def export_encoding_plan(self, plan: EncodingPlan) -> str:
        """Export the encoding plan to disk."""
        return self.registry.export_plan(plan)

    def load_encoding_plan(self, plan_path: Path | str | None = None) -> EncodingPlan:
        """Load an encoding plan from disk."""
        path = Path(plan_path or Path(self.config.project_root) / self.ENCODING_PLAN_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Encoding plan not found: {path}")

        plan = EncodingPlan()
        with path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = pd.read_csv(csvfile)
            for _, row in reader.iterrows():
                plan.entries.append(
                    EncodingPlanEntry(
                        feature_name=row["feature_name"],
                        feature_type=row["feature_type"],
                        encoding_strategy=row["encoding_strategy"],
                    )
                )
        return plan

    def fit(self, X_train: pd.DataFrame, plan: EncodingPlan | None = None) -> None:
        """Fit the encoders on the training set using the encoding plan."""
        if plan is None:
            plan = self.load_encoding_plan()

        self.onehot_features = [
            entry.feature_name
            for entry in plan.entries
            if entry.encoding_strategy == self.registry.ONEHOT_ENCODING
            and entry.feature_name in X_train.columns
        ]
        self.ordinal_features = [
            entry.feature_name
            for entry in plan.entries
            if entry.encoding_strategy == self.registry.ORDINAL_ENCODING
            and entry.feature_name in X_train.columns
        ]
        self.non_encoded_features = [
            column
            for column in X_train.columns
            if column not in self.onehot_features + self.ordinal_features
        ]

        self.logger.info(
            "Encoding started",
            onehot_features=len(self.onehot_features),
            ordinal_features=len(self.ordinal_features),
        )

        if self.onehot_features:
            self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_encoder.fit(X_train[self.onehot_features].astype(str))
            self.logger.info("OneHot fitted", features=len(self.onehot_features))

        if self.ordinal_features:
            self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.ordinal_encoder.fit(X_train[self.ordinal_features].astype(str))
            self.logger.info("Ordinal fitted", features=len(self.ordinal_features))

    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform training and testing datasets using the fitted encoders."""
        if self.onehot_encoder is None and self.ordinal_encoder is None:
            raise ValueError("Encoders have not been fitted.")

        X_train_before = X_train.shape
        X_test_before = X_test.shape

        encoded_X_train = self._transform_dataframe(X_train)
        encoded_X_test = self._transform_dataframe(X_test)

        self.logger.info(
            "Training transformed",
            shape_before=X_train_before,
            shape_after=encoded_X_train.shape,
        )
        self.logger.info(
            "Testing transformed",
            shape_before=X_test_before,
            shape_after=encoded_X_test.shape,
        )

        self.logger.info(
            "Encoding completed",
            onehot_features=len(self.onehot_features),
            ordinal_features=len(self.ordinal_features),
        )
        return encoded_X_train, encoded_X_test

    def _transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        X_non_encoded = X[self.non_encoded_features].copy()
        result_frames: list[pd.DataFrame] = [X_non_encoded]

        if self.ordinal_features and self.ordinal_encoder is not None:
            ordinal_values = self.ordinal_encoder.transform(X[self.ordinal_features].astype(str))
            ordinal_df = pd.DataFrame(
                ordinal_values,
                columns=self.ordinal_features,
                index=X.index,
            )
            result_frames.append(ordinal_df)

        if self.onehot_features and self.onehot_encoder is not None:
            onehot_values = self.onehot_encoder.transform(X[self.onehot_features].astype(str))
            onehot_columns = self.onehot_encoder.get_feature_names_out(self.onehot_features)
            onehot_df = pd.DataFrame(onehot_values, columns=onehot_columns, index=X.index)
            result_frames.append(onehot_df)

        encoded = pd.concat(result_frames, axis=1)
        return encoded

    def export_encoders(self) -> tuple[str, str]:
        """Persist the fitted encoders to disk for inference."""
        output_dir = Path(self.config.project_root) / self.ENCODER_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        onehot_path = output_dir / self.ONEHOT_ENCODER_FILE
        ordinal_path = output_dir / self.ORDINAL_ENCODER_FILE

        if self.onehot_encoder is not None:
            dump(self.onehot_encoder, onehot_path)
        if self.ordinal_encoder is not None:
            dump(self.ordinal_encoder, ordinal_path)

        self.logger.info(
            "Encoders exported",
            onehot_path=str(onehot_path),
            ordinal_path=str(ordinal_path),
        )
        return str(onehot_path), str(ordinal_path)
