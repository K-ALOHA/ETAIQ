"""Feature scaling module for ETAIQ preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .config import FeatureEngineeringConfig
from .logging_config import FeatureEngineeringLogger
from .utils import ensure_directory


class ScalingEngine:
    """Fit and apply scaling to continuous numerical features."""

    SCALER_DIR = Path("ml") / "models" / "preprocessing"
    SCALER_FILE = "standard_scaler.pkl"
    ENCODING_PLAN_FILE = Path("ml") / "data" / "features" / "encoding_plan.csv"

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.scaler: StandardScaler | None = None
        self.onehot_features: list[str] = []
        self.onehot_columns: list[str] = []
        self.ordinal_features: list[str] = []
        self.boolean_columns: list[str] = []
        self.identifier_columns: list[str] = []
        self.continuous_columns: list[str] = []

    def load_encoding_plan(self, plan_path: Path | str | None = None) -> pd.DataFrame:
        """Load the encoding plan from disk."""
        path = Path(plan_path or Path(self.config.project_root) / self.ENCODING_PLAN_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Encoding plan not found: {path}")
        return pd.read_csv(path)

    def fit(self, encoded_X_train: pd.DataFrame, plan_path: Path | str | None = None) -> None:
        """Fit the scaler on continuous numerical features from the encoded training set."""
        plan = self.load_encoding_plan(plan_path)
        self.onehot_features = plan.loc[plan["encoding_strategy"] == "OneHot Encoding", "feature_name"].tolist()
        self.ordinal_features = plan.loc[plan["encoding_strategy"] == "Ordinal Encoding", "feature_name"].tolist()

        self.boolean_columns = [column for column in encoded_X_train.columns if is_bool_dtype(encoded_X_train[column].dtype)]
        self.identifier_columns = [column for column in encoded_X_train.columns if self._is_identifier(column)]
        self.onehot_columns = self._resolve_onehot_columns(encoded_X_train.columns)

        self.continuous_columns = [
            column
            for column in encoded_X_train.columns
            if is_numeric_dtype(encoded_X_train[column].dtype)
            and column not in self.boolean_columns
            and column not in self.identifier_columns
            and column not in self.onehot_columns
            and column not in self.ordinal_features
        ]

        self.logger.info(
            "Scaling started",
            numerical_features=len(self.continuous_columns),
        )
        self.logger.info(
            "Numerical features detected",
            continuous_features=self.continuous_columns,
        )

        if self.continuous_columns:
            self.scaler = StandardScaler()
            self.scaler.fit(encoded_X_train[self.continuous_columns])
            self.logger.info("Scaler fitted", features=len(self.continuous_columns))

    def transform(
        self,
        encoded_X_train: pd.DataFrame,
        encoded_X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform encoded training and testing dataframes using the fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")

        train_before = encoded_X_train.shape
        test_before = encoded_X_test.shape

        scaled_X_train = self._scale_dataframe(encoded_X_train)
        scaled_X_test = self._scale_dataframe(encoded_X_test)

        self.logger.info(
            "Training transformed",
            shape_before=train_before,
            shape_after=scaled_X_train.shape,
        )
        self.logger.info(
            "Testing transformed",
            shape_before=test_before,
            shape_after=scaled_X_test.shape,
        )

        return scaled_X_train, scaled_X_test

    def export_scaler(self) -> str:
        """Persist the fitted scaler to disk."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")

        scaler_path = Path(self.config.project_root) / self.SCALER_DIR / self.SCALER_FILE
        ensure_directory(scaler_path.parent)
        dump(self.scaler, scaler_path)

        self.logger.info("Scaler exported", path=str(scaler_path))
        return str(scaler_path)

    def _resolve_onehot_columns(self, columns: list[str]) -> list[str]:
        result: list[str] = []
        normalized_columns = [str(column) for column in columns]
        for feature_name in self.onehot_features:
            prefix = f"{str(feature_name)}_"
            result.extend([column for column in normalized_columns if str(column).startswith(prefix)])
        return result

    def _scale_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if self.continuous_columns:
            scaled_values = self.scaler.transform(result[self.continuous_columns])
            scaled_df = pd.DataFrame(
                scaled_values,
                columns=self.continuous_columns,
                index=result.index,
            )
            for column in self.continuous_columns:
                result[column] = scaled_df[column].values
        return result

    def _is_identifier(self, column: str) -> bool:
        name = str(column).lower()
        return name == "id" or name.endswith("_id") or "_id" in name
