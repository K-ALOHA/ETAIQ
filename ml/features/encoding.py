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
        self.high_cardinality_skipped_features: list[str] = []

    def prepare_encoding_plan(
        self,
        feature_registry: FeatureRegistry,
        X_train: pd.DataFrame | None = None,
    ) -> EncodingPlan:
        """Inspect the feature registry and generate an encoding plan."""
        plan = self.registry.create_plan(feature_registry, X_train=X_train)
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

        self.high_cardinality_skipped_features = [
            entry.feature_name
            for entry in plan.entries
            if entry.encoding_strategy == self.registry.SKIPPED_HIGH_CARDINALITY
        ]

        self.logger.info("Feature count before encoding", feature_count=len(X_train.columns))

        self.logger.info(
            "Encoding started",
            onehot_features=len(self.onehot_features),
            ordinal_features=len(self.ordinal_features),
            high_cardinality_skipped=len(self.high_cardinality_skipped_features),
        )
        if self.high_cardinality_skipped_features:
            self.logger.info(
                "High-cardinality skipped features",
                features=self.high_cardinality_skipped_features,
            )

        self._print_encoding_summary(
            feature_count_before=len(X_train.columns),
            feature_count_after=len(X_train.columns),
            onehot_features=self.onehot_features,
            ordinal_features=self.ordinal_features,
            skipped_features=self.high_cardinality_skipped_features,
        )

        for feature_name in self.high_cardinality_skipped_features:
            reason = self._get_high_cardinality_reason(feature_name, X_train)
            self.logger.info(
                "High-cardinality column skipped",
                feature_name=feature_name,
                reason=reason,
            )

        if self.onehot_features:
            assert all(feature not in self.high_cardinality_skipped_features for feature in self.onehot_features)
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
            self.logger.info("Encoding skipped; no encoders were configured")
            return X_train.copy(), X_test.copy()

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
        self._print_encoding_summary(
            feature_count_before=X_train.shape[1],
            feature_count_after=encoded_X_train.shape[1],
            onehot_features=self.onehot_features,
            ordinal_features=self.ordinal_features,
            skipped_features=self.high_cardinality_skipped_features,
        )
        return encoded_X_train, encoded_X_test

    def transform_single(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a single dataframe using the fitted encoders."""
        if self.onehot_encoder is None and self.ordinal_encoder is None:
            self.logger.info("Encoding skipped; no encoders were configured")
            return X.copy()

        return self._transform_dataframe(X)

    def _transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        # Only keep columns that actually exist in the incoming dataframe
        non_encoded_present = [c for c in self.non_encoded_features if c in X.columns]
        missing = [c for c in self.non_encoded_features if c not in X.columns]
        if missing:
            self.logger.info("Non-encoded features missing in input; skipping missing columns", missing_features=missing)
        X_non_encoded = X[non_encoded_present].copy()
        result_frames: list[pd.DataFrame] = [X_non_encoded]
        if self.ordinal_features and self.ordinal_encoder is not None:
            ordinal_present = [c for c in self.ordinal_features if c in X.columns]
            if ordinal_present:
                ordinal_values = self.ordinal_encoder.transform(X[ordinal_present].astype(str))
                ordinal_df = pd.DataFrame(
                    ordinal_values,
                    columns=ordinal_present,
                    index=X.index,
                )
                result_frames.append(ordinal_df)

        if self.onehot_features and self.onehot_encoder is not None:
            onehot_present = [c for c in self.onehot_features if c in X.columns]
            if onehot_present:
                onehot_values = self.onehot_encoder.transform(X[onehot_present].astype(str))
                onehot_columns = self.onehot_encoder.get_feature_names_out(onehot_present)
                onehot_df = pd.DataFrame(onehot_values, columns=onehot_columns, index=X.index)
                result_frames.append(onehot_df)

        encoded = pd.concat(result_frames, axis=1)
        return encoded

    def _get_high_cardinality_reason(self, feature_name: str, X_train: pd.DataFrame) -> str | None:
        if feature_name not in X_train.columns:
            return None
        total_rows = len(X_train)
        if total_rows == 0:
            return None
        unique_values = int(X_train[feature_name].nunique(dropna=False))
        reasons: list[str] = []
        if unique_values > 100:
            reasons.append(f"unique_values={unique_values} > 100")
        if (unique_values / total_rows) > 0.10:
            reasons.append(f"ratio={unique_values / total_rows:.2%} > 10%")
        return "; ".join(reasons) if reasons else None

    def _print_encoding_summary(
        self,
        feature_count_before: int,
        feature_count_after: int,
        onehot_features: list[str],
        ordinal_features: list[str],
        skipped_features: list[str],
    ) -> None:
        print("=" * 40)
        print("Encoding Summary")
        print("=" * 40)
        print(f"OneHot features: {len(onehot_features)}")
        print(f"Ordinal features: {len(ordinal_features)}")
        print(f"High-cardinality skipped features: {len(skipped_features)}")
        print(f"Feature count before encoding: {feature_count_before}")
        print(f"Feature count after encoding: {feature_count_after}")
        print("=" * 40)

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
