"""Feature selection module for ETAIQ preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype

from .config import FeatureEngineeringConfig
from .logging_config import FeatureEngineeringLogger
from .utils import ensure_directory


class FeatureSelectionEngine:
    """Select a stable subset of features for model training."""

    FEATURE_IMPORTANCE_FILE = Path("ml") / "data" / "features" / "feature_importance.csv"
    SELECTED_FEATURES_FILE = Path("ml") / "data" / "features" / "selected_features.csv"
    MODEL_DIR = Path("ml") / "models" / "preprocessing"
    MODEL_FILE = "feature_selection_model.pkl"

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.model: RandomForestRegressor | None = None

    def select_features(
        self,
        scaled_X_train: pd.DataFrame,
        scaled_X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform feature selection and export selection artifacts."""
        self.logger.info("Feature selection started")

        original_features = list(scaled_X_train.columns)
        current_train = scaled_X_train.copy()

        current_train = self._drop_non_numeric_columns(current_train)

        current_train, constant_features = self._remove_constant_features(current_train)
        constant_removed = len(constant_features)
        current_train, correlated_features = self._remove_correlated_features(current_train, threshold=0.95)
        correlated_removed = len(correlated_features)

        if current_train.empty:
            self.logger.info("No feature columns remain after preprocessing; skipping feature selection")
            selected_feature_names = list(scaled_X_train.columns)
            selected_X_train = scaled_X_train[selected_feature_names].copy()
            selected_X_test = scaled_X_test[selected_feature_names].copy()
            importance_df = pd.DataFrame(
                {
                    "feature_name": selected_feature_names,
                    "importance": 0.0,
                }
            )
            importance_df["rank"] = range(1, len(importance_df) + 1)
            self._export_feature_importance(importance_df)
            self._export_selected_features(importance_df)
            self._export_model()
            self._print_summary(
                original_features=len(original_features),
                constant_removed=constant_removed,
                correlated_removed=correlated_removed,
                selected_features=len(selected_feature_names),
                top_features=importance_df["feature_name"].head(20).tolist(),
            )
            return selected_X_train, selected_X_test

        self.model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.model.fit(current_train, y_train)
        self.logger.info("Feature importance calculated", features=current_train.shape[1])

        importance_df = pd.DataFrame(
            {
                "feature_name": current_train.columns,
                "importance": self.model.feature_importances_,
            }
        )
        importance_df.sort_values("importance", ascending=False, inplace=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        selected_feature_names = importance_df["feature_name"].tolist()
        selected_X_train = scaled_X_train[selected_feature_names].copy()
        selected_X_test = scaled_X_test[selected_feature_names].copy()

        self._export_feature_importance(importance_df)
        self._export_selected_features(importance_df)
        self._export_model()

        self._print_summary(
            original_features=len(original_features),
            constant_removed=constant_removed,
            correlated_removed=correlated_removed,
            selected_features=len(selected_feature_names),
            top_features=importance_df["feature_name"].head(20).tolist(),
        )

        return selected_X_train, selected_X_test

    def _remove_constant_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        constant_features = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
        if constant_features:
            self.logger.info("Constant features removed", features=constant_features)
            X = X.drop(columns=constant_features)
        return X, constant_features

    def _drop_non_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        non_numeric = [column for column in X.columns if not is_numeric_dtype(X[column].dtype)]
        if non_numeric:
            self.logger.info("Non-numeric features dropped before selection", features=non_numeric)
            X = X.drop(columns=non_numeric)
        return X

    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
        numeric = X.select_dtypes(include=[np.number])
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]
        if correlated_features:
            self.logger.info("Correlated features removed", features=correlated_features)
            X = X.drop(columns=correlated_features)
        return X, correlated_features

    def _export_feature_importance(self, importance_df: pd.DataFrame) -> str:
        output_path = Path(self.config.project_root) / self.FEATURE_IMPORTANCE_FILE
        ensure_directory(output_path.parent)
        importance_df.to_csv(output_path, index=False)
        self.logger.info("Feature importance exported", path=str(output_path))
        return str(output_path)

    def _export_selected_features(self, importance_df: pd.DataFrame) -> str:
        output_path = Path(self.config.project_root) / self.SELECTED_FEATURES_FILE
        ensure_directory(output_path.parent)
        selected_df = importance_df[["feature_name", "rank", "importance"]].copy()
        selected_df.to_csv(output_path, index=False)
        self.logger.info("Selected features exported", path=str(output_path))
        return str(output_path)

    def _export_model(self) -> str | None:
        if self.model is None:
            return ""
        output_dir = Path(self.config.project_root) / self.MODEL_DIR
        ensure_directory(output_dir)
        model_path = output_dir / self.MODEL_FILE
        dump(self.model, model_path)
        self.logger.info("Feature selection model exported", path=str(model_path))
        return str(model_path)

    def _print_summary(
        self,
        original_features: int,
        constant_removed: int,
        correlated_removed: int,
        selected_features: int,
        top_features: list[str],
    ) -> None:
        print("=" * 40)
        print("Feature Selection Summary")
        print("=" * 40)
        print(f"Original features: {original_features}")
        print(f"Constant removed: {constant_removed}")
        print(f"Correlated removed: {correlated_removed}")
        print(f"Final selected features: {selected_features}")
        print("Top 20 Most Important Features:")
        for feature in top_features:
            print(f"- {feature}")
        print("=" * 40)
