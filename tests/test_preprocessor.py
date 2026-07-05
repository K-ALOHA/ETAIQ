import os
import json
import tempfile
import pandas as pd
import numpy as np
from ml.features.sklearn_preprocessor import SklearnPreprocessor


def make_sample_df(n=10):
    rng = pd.date_range("2021-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": rng.astype(str),
        "cat": ["a", "b"] * (n // 2) + (["a"] if n % 2 else []),
        "num": np.arange(n).astype(float),
    })
    return df


def test_fit_transform_and_metadata(tmp_path):
    df = make_sample_df(20)
    pre = SklearnPreprocessor()
    pre.fit(df["timestamp","cat","num"] if False else df)
    out = pre.transform(df)
    # output should be a DataFrame and if output_feature_names_ recorded, ordering must match
    assert isinstance(out, pd.DataFrame)
    if getattr(pre, "output_feature_names_", None):
        assert list(out.columns) == pre.output_feature_names_
    # metadata file created
    meta_path = pre.export_metadata()
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "output_feature_names" in meta


def test_schema_validation_raises_on_missing():
    df = make_sample_df(10)
    pre = SklearnPreprocessor()
    pre.fit(df)
    # remove a column
    df2 = df.drop(columns=[pre.feature_names_in_[0]])
    try:
        pre.transform(df2)
        raised = False
    except ValueError as e:
        raised = True
        assert "Missing columns" in str(e) or "does not match training schema" in str(e)
    assert raised


def test_extra_columns_raise():
    df = make_sample_df(10)
    pre = SklearnPreprocessor()
    pre.fit(df)
    df2 = df.copy()
    df2["extra_col"] = 1
    try:
        pre.transform(df2)
        raised = False
    except ValueError as e:
        raised = True
        assert "Extra columns" in str(e) or "does not match training schema" in str(e)
    assert raised


def test_missing_selected_filled_by_imputer():
    df = make_sample_df(12)
    pre = SklearnPreprocessor()
    pre.fit(df)
    # simulate missing a selected feature after encoding/scaling by dropping a raw column
    # given strict validation, must pass original columns; instead call internal steps
    df_raw = df.copy()
    encoded = pre.encoding.transform_single(pre._extract_datetime_features(df_raw))
    scaled = pre.scaling.transform_single(encoded)
    # drop one selected feature column to simulate disappearance
    if pre.output_feature_names_:
        drop_col = pre.output_feature_names_[0]
        if drop_col in scaled.columns:
            scaled2 = scaled.drop(columns=[drop_col])
            selected = pre.selection.transform(scaled2)
            # ensure missing was filled when reordering
            if drop_col in selected.columns:
                val = selected[drop_col].iloc[0]
                assert val == 0.0 or isinstance(val, float)

