"""A scikit-learn compatible preprocessor that wraps existing encoding, scaling, and selection engines."""

from __future__ import annotations

from typing import Any
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from .encoding import EncodingEngine
from .scaling import ScalingEngine
from .selection import FeatureSelectionEngine

logger = logging.getLogger(__name__)


class SklearnPreprocessor(BaseEstimator, TransformerMixin):
    """Wrapper transformer that fits encoding, scaling, and feature selection.

    The transformer exposes `selected_features_` after fitting for downstream use.
    """

    def __init__(self) -> None:
        self.encoding = EncodingEngine()
        self.scaling = ScalingEngine()
        self.selection = FeatureSelectionEngine()
        self.selected_features_: list[str] = []
        self.output_feature_names_: list[str] = []
        self.datetime_columns_: list[str] = []
        self.raw_numeric_columns_: list[str] = []
        self.imputer: SimpleImputer | None = None
        self.imputer_statistics_: dict[str, float] = {}
        self.preprocessing_version: int = 1
        self.metadata_path = Path("ml") / "artifacts" / "preprocessing" / "preprocessor_metadata.json"

    def _prepare_df(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        if hasattr(self, "feature_names_in_") and df.shape[1] == len(getattr(self, "feature_names_in_", [])):
            feature_names = list(self.feature_names_in_)
            default_index_names = list(range(df.shape[1]))
            normalized_columns = [int(str(c)) if str(c).isdigit() else c for c in df.columns]
            if normalized_columns == default_index_names:
                df.columns = feature_names
        df.columns = [str(c) for c in df.columns]
        return df

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert any parseable datetime-like columns into numeric features."""
        result = df.copy()
        datetimes: list[str] = []
        # detect datetime-like cols first
        for col in list(result.columns):
            if pd.api.types.is_datetime64_any_dtype(result[col].dtype):
                datetimes.append(col)
                continue
            if pd.api.types.is_numeric_dtype(result[col].dtype) or pd.api.types.is_bool_dtype(result[col].dtype):
                continue
            try:
                parsed = pd.to_datetime(result[col], errors="coerce", utc=True)
                non_null = parsed.notna().sum()
                if non_null > 0 and (non_null / len(parsed)) > 0.5:
                    result[col] = parsed
                    datetimes.append(col)
            except Exception:
                continue

        # build derived columns in batch to avoid fragmentation
        derived: dict[str, list[Any]] = {}
        for col in datetimes:
            ts = result[col]
            derived[f"{col}_epoch"] = (ts.astype("int64") // 10 ** 9).fillna(0).astype("int64")
            derived[f"{col}_hour"] = ts.dt.hour.fillna(-1).astype(int)
            derived[f"{col}_minute"] = ts.dt.minute.fillna(-1).astype(int)
            derived[f"{col}_dayofweek"] = ts.dt.dayofweek.fillna(-1).astype(int)
            derived[f"{col}_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)

        if derived:
            derived_df = pd.DataFrame(derived, index=result.index)
            # drop original datetime cols then concat derived together once
            result = result.drop(columns=datetimes)
            result = pd.concat([result, derived_df], axis=1)

        self.datetime_columns_ = datetimes
        return result

    def fit(self, X: Any, y: Any = None) -> "SklearnPreprocessor":
        df = self._prepare_df(X)
        df = self._extract_datetime_features(df)

        # Build an encoding plan: numerics -> no encoding, objects -> ordinal/onehot heuristics
        from .encoding_registry import EncodingPlan, EncodingPlanEntry

        plan = EncodingPlan()
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                plan.add_entry(str(col), "Numerical", "No Encoding")
            else:
                # choose encoding by cardinality
                uniques = df[col].nunique(dropna=False)
                if uniques <= 10:
                    plan.add_entry(str(col), "Categorical", self.encoding.registry.ONEHOT_ENCODING)
                else:
                    plan.add_entry(str(col), "Categorical", self.encoding.registry.ORDINAL_ENCODING)

        # Persist an encoding plan for downstream scaling and inference consistency.
        try:
            plan_path = self.encoding.export_encoding_plan(plan)
        except Exception:
            plan_path = None

        # Fit encoders
        self.encoding.fit(df, plan=plan)
        encoded, _ = self.encoding.transform(df, df)

        # Fit scaler
        self.scaling.fit(encoded, plan_path=plan_path)
        scaled = self.scaling.transform_single(encoded)

        # Impute missing numeric values to avoid downstream estimator failures
        numeric_cols = [c for c in scaled.columns if pd.api.types.is_numeric_dtype(scaled[c].dtype)]
        self.numeric_columns_ = numeric_cols
        if numeric_cols:
            self.imputer = SimpleImputer(strategy="median")
            scaled[numeric_cols] = self.imputer.fit_transform(scaled[numeric_cols])
            # store statistics as dict for persistence
            try:
                stats = dict(zip(numeric_cols, [float(x) for x in self.imputer.statistics_]))
            except Exception:
                stats = {}
            self.imputer_statistics_ = stats

        # Fit selector
        if y is None:
            # create dummy target if not provided
            import numpy as _np

            y_ser = pd.Series(_np.arange(len(scaled), dtype=float))
        else:
            y_ser = pd.Series(y).reset_index(drop=True)

        self.selection.fit(scaled, y_ser)
        self.selected_features_ = getattr(self.selection, "selected_feature_names", list(scaled.columns))
        # Record final output feature ordering for the estimator
        self.output_feature_names_ = list(self.selected_features_)
        # record original input feature names for transform-time reconstruction
        self.feature_names_in_ = [str(c) for c in pd.DataFrame(X).columns]
        self.raw_numeric_columns_ = [str(c) for c in pd.DataFrame(X).select_dtypes(include=["number"]).columns]
        # export preprocessing artifacts (encoders, scaler, selection model, metadata)
        try:
            self.encoding.export_encoders()
        except Exception as e:
            logger.warning("Failed to export encoders", error=str(e))
        try:
            self.scaling.export_scaler()
        except Exception as e:
            logger.warning("Failed to export scaler", error=str(e))
        try:
            # selection exports internally during fit
            pass
        except Exception as e:
            logger.warning("Selection export failed", error=str(e))
        try:
            self.export_metadata()
        except Exception as e:
            logger.warning("Failed to export metadata", error=str(e))
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        # Reconstruct incoming DataFrame using recorded feature names when possible
        df = self._prepare_df(X)
        # Validate that incoming raw schema exactly matches training schema
        incoming_cols = [str(c) for c in df.columns]
        if hasattr(self, "feature_names_in_"):
            trained_cols = list(self.feature_names_in_)
            missing = [c for c in trained_cols if c not in incoming_cols]
            extra = [c for c in incoming_cols if c not in trained_cols]
            if missing or extra:
                raise ValueError(
                    f"Input schema does not match training schema. Missing columns: {missing}. Extra columns: {extra}."
                )
        df = self._extract_datetime_features(df)
        # Preserve numeric semantics for raw numeric columns when raw input arrives as strings
        if hasattr(self, "raw_numeric_columns_") and self.raw_numeric_columns_:
            for col in self.raw_numeric_columns_:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        encoded = self.encoding.transform_single(df)
        scaled = self.scaling.transform_single(encoded)
        # Apply imputation using fitted imputer
        if self.imputer is not None:
            numeric_cols = [c for c in scaled.columns if pd.api.types.is_numeric_dtype(scaled[c].dtype)]
            if numeric_cols:
                scaled[numeric_cols] = self.imputer.transform(scaled[numeric_cols])

        selected = self.selection.transform(scaled)

        # Ensure deterministic output: all output_feature_names_ present and in order
        if hasattr(self, "output_feature_names_") and self.output_feature_names_:
            desired = list(self.output_feature_names_)
            missing = [c for c in desired if c not in selected.columns]
            extra = [c for c in selected.columns if c not in desired]
            # Fill any missing numeric features using imputer statistics, else fill with zeros
            for c in missing:
                if c in self.imputer_statistics_:
                    selected[c] = float(self.imputer_statistics_.get(c, 0.0))
                else:
                    selected[c] = 0.0
            # Drop unexpected extra columns to match training output exactly
            if extra:
                selected = selected.drop(columns=extra)
            # Reorder to desired ordering
            selected = selected[desired].copy()
            # Final validation
            if list(selected.columns) != desired:
                raise ValueError(f"Transformed output columns do not match training output schema: expected {desired}, got {list(selected.columns)}")

        return selected

    def fit_transform(self, X: Any, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def export_metadata(self) -> str:
        """Persist metadata about preprocessing so inference can validate and reproduce."""
        meta = {
            "preprocessing_version": int(self.preprocessing_version),
            "feature_names_in": list(getattr(self, "feature_names_in_", [])),
            "datetime_columns": list(getattr(self, "datetime_columns_", [])),
            "numeric_columns": list(getattr(self, "numeric_columns_", [])),
            "raw_numeric_columns": list(getattr(self, "raw_numeric_columns_", [])),
            "imputer_statistics": getattr(self, "imputer_statistics_", {}),
            "output_feature_names": list(getattr(self, "output_feature_names_", [])),
        }
        path = Path(self.metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return str(path)

    def load_metadata(self, path: str | Path | None = None) -> None:
        p = Path(path or self.metadata_path)
        if not p.exists():
            raise FileNotFoundError(f"Preprocessor metadata not found: {p}")
        with p.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.preprocessing_version = int(meta.get("preprocessing_version", 1))
        self.feature_names_in_ = meta.get("feature_names_in", [])
        self.datetime_columns_ = meta.get("datetime_columns", [])
        self.raw_numeric_columns_ = meta.get("raw_numeric_columns", [])
        self.numeric_columns_ = meta.get("numeric_columns", [])
        self.imputer_statistics_ = meta.get("imputer_statistics", {})
        self.output_feature_names_ = meta.get("output_feature_names", [])
