"""Unit and integration tests for the ETAIQ Cleaning Execution Engine.

Tests verify that the cleaning engine performs ONLY data cleaning operations:
- Remove duplicates
- Impute missing values
- Fix data types
- Clean timestamps
- Validate GPS coordinates
- Validate foreign keys
- Remove outliers
- Drop columns when necessary

Tests do NOT include ML feature preparation (that belongs in a separate
transformation module after cleaning):
- Standardization/Normalization (feature scaling)
- Encoding (categorical to numeric)
- Column renaming
- Schema transformation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.cleaning.cleaning_engine import CleaningEngine
from ml.cleaning.config import CleaningConfig
from ml.cleaning.datatype_cleaner import DatatypeCleaner
from ml.cleaning.duplicate_handler import DuplicateHandler
from ml.cleaning.gps_cleaner import GpsCleaner
from ml.cleaning.imputation import ImputationHandler
from ml.cleaning.integrity_cleaner import IntegrityCleaner
from ml.cleaning.main import main
from ml.cleaning.outlier_handler import OutlierHandler
from ml.cleaning.rollback import calculate_checksum
from ml.cleaning.timestamp_cleaner import TimestampCleaner


@pytest.fixture
def mock_raw_and_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Sets up a mock directories containing raw data and manifest."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    raw_dir.mkdir()
    processed_dir.mkdir()
    reports_dir.mkdir()

    # Create raw datasets
    restaurants_df = pd.DataFrame(
        {
            "restaurant_id": ["1", "2", "2"],  # Duplicate row
            "restaurant_name": ["Spice Diner", "Urban Eats", "Urban Eats"],
            "latitude": [12.97, 12.98, 12.98],
            "longitude": [77.59, 77.60, 77.60],
            "city": ["Bangalore", "Bangalore", "Bangalore"],
            "is_active": [True, True, True],
        }
    )
    riders_df = pd.DataFrame(
        {
            "rider_id": ["D001", "D002"],
            "rider_name": ["Rider 1", "Rider 2"],
            "latitude": [12.99, 999.0],  # Out of bounds GPS coordinate
            "longitude": [77.58, 77.58],
            "vehicle_type": ["bike", "scooter"],
            "status": ["active", "active"],
        }
    )
    orders_df = pd.DataFrame(
        {
            "order_id": ["O001", "O002", "O003"],
            "restaurant_id": ["1", "999", "2"],  # "999" is an orphan foreign key
            "rider_id": ["D001", "D002", "D001"],
            "order_timestamp": ["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00"],
            "pickup_timestamp": ["2026-01-01 10:10", "2026-01-01 11:10", "2026-01-01 12:10"],
            "delivery_timestamp": ["2026-01-01 10:35", "2026-01-01 11:35", "2026-01-01 12:35"],
            "delivery_time_minutes": [25.0, 25.0, -99.0],  # Negative outlier
            "customer_latitude": [12.99, 12.99, 12.99],
            "customer_longitude": [77.58, 77.58, 77.58],
        }
    )

    restaurants_df.to_csv(raw_dir / "restaurants.csv", index=False)
    riders_df.to_csv(raw_dir / "riders.csv", index=False)
    orders_df.to_csv(raw_dir / "orders.csv", index=False)

    # Create approval manifest
    manifest = {
        "manifest": [
            {
                "decision_id": "DEC-001",
                "dataset_id": "restaurants",
                "column": None,
                "action": "REMOVE_DUPLICATES",
                "priority": "HIGH",
                "confidence": 1.0,
                "rationale": "Remove duplicates",
                "status": "APPROVED",
            },
            {
                "decision_id": "DEC-002",
                "dataset_id": "orders",
                "column": "delivery_time_minutes",
                "action": "REMOVE_OUTLIERS",
                "priority": "HIGH",
                "confidence": 0.95,
                "rationale": "Remove negative delivery times",
                "status": "APPROVED",
            },
            {
                "decision_id": "DEC-003",
                "dataset_id": "orders",
                "column": "restaurant_id",
                "action": "DROP",
                "priority": "MEDIUM",
                "confidence": 0.90,
                "rationale": "Drop rows with orphan foreign keys",
                "status": "APPROVED",
            },
            {
                "decision_id": "DEC-004",
                "dataset_id": "riders",
                "column": "latitude",
                "action": "FLAG",
                "priority": "MEDIUM",
                "confidence": 0.95,
                "rationale": "Clean coordinate limits",
                "status": "PENDING_APPROVAL",  # Test filtering behavior
            },
        ]
    }
    (reports_dir / "approval_manifest.json").write_text(json.dumps(manifest))

    # Mock estimated_quality.json
    est_quality = {
        "metrics": {
            "data_quality": {"baseline": 66.62, "target": 98.50, "delta": 31.88},
            "completeness": {"baseline": 99.31, "target": 100.00, "delta": 0.69},
            "consistency": {"baseline": 25.00, "target": 100.00, "delta": 75.00},
            "integrity": {"baseline": 0.00, "target": 98.00, "delta": 98.00},
            "model_reliability": {"baseline": 35.00, "target": 100.00, "delta": 65.00},
        }
    }
    (reports_dir / "estimated_quality.json").write_text(json.dumps(est_quality))

    return raw_dir, processed_dir, reports_dir


def test_duplicate_handler() -> None:
    """Test duplicate handler removes duplicates correctly."""
    df = pd.DataFrame({"id": [1, 2, 2], "val": ["a", "b", "b"]})
    cleaned, result = DuplicateHandler().execute(df, "test")
    assert len(cleaned) == 2
    assert result.success
    assert result.details["dropped_count"] == 1


def test_imputation_handler() -> None:
    """Test imputation handler fills null values based on datatype."""
    # Numeric imputation with median
    df_num = pd.DataFrame({"val": [10.0, 20.0, None]})
    cleaned_num, res_num = ImputationHandler().execute(df_num, "test", "val")
    assert cleaned_num["val"].isna().sum() == 0
    assert cleaned_num["val"].iloc[2] == 15.0  # Median of 10 & 20

    # Categorical imputation with mode
    df_cat = pd.DataFrame({"val": ["apple", "apple", None]})
    cleaned_cat, res_cat = ImputationHandler().execute(df_cat, "test", "val")
    assert cleaned_cat["val"].isna().sum() == 0
    assert cleaned_cat["val"].iloc[2] == "apple"  # Mode


def test_outlier_handler() -> None:
    """Test outlier handler filters out negative and extreme outliers."""
    df = pd.DataFrame({"delivery_time_minutes": [20.0, 30.0, -10.0]})  # Negative values are dropped
    cleaned, result = OutlierHandler().execute(df, "test", "delivery_time_minutes")
    assert len(cleaned) == 2
    assert result.details["removed_count"] == 1


def test_datatype_cleaner() -> None:
    """Test standardizing datatypes."""
    df = pd.DataFrame({"id": ["1", "2"], "val": ["1.0", "0.0"]})
    cleaned, result = DatatypeCleaner().execute(df, "test", "val", "bool")
    assert cleaned["val"].dtype == "boolean"
    assert cleaned["val"].iloc[0]


def test_gps_cleaner() -> None:
    """Test coordinates out of bounds are dropped."""
    df = pd.DataFrame({"latitude": [12.97, 999.0]})
    cleaned, result = GpsCleaner().execute(df, "test", "latitude")
    assert len(cleaned) == 1
    assert result.details["removed_count"] == 1


def test_timestamp_cleaner() -> None:
    """Test timestamp values formatting."""
    df = pd.DataFrame({"timestamp": ["18/05/2026 22:56"]})
    cleaned, result = TimestampCleaner().execute(df, "test", "timestamp")
    assert cleaned["timestamp"].iloc[0] == "2026-05-18 22:56:00"


def test_integrity_cleaner() -> None:
    """Test foreign key orphans validation."""
    df = pd.DataFrame({"restaurant_id": ["1", "2", "3"]})
    ref_keys = {"1", "2"}
    cleaned, result = IntegrityCleaner().execute(df, "test", "restaurant_id", "FLAG", reference_keys=ref_keys)
    assert cleaned["restaurant_id_is_orphan"].iloc[2]


def test_schema_preservation(mock_raw_and_manifest: tuple[Path, Path, Path]) -> None:
    """Verify that the cleaning engine preserves the original schema.
    
    The cleaned dataset must have:
    - Same columns as raw dataset
    - Same column order
    - Same column names (NO renaming)
    - Only repaired data values (NO schema transformation)
    """
    raw_dir, proc_dir, rep_dir = mock_raw_and_manifest

    config = CleaningConfig(raw_dir=raw_dir, processed_dir=proc_dir, reports_dir=rep_dir)
    engine = CleaningEngine(config=config)

    # Run cleaning
    engine.run(force_approve_all=True)

    # Load raw and processed datasets
    df_raw_orders = pd.read_csv(raw_dir / "orders.csv")
    df_proc_orders = pd.read_csv(proc_dir / "orders.csv")
    
    df_raw_restaurants = pd.read_csv(raw_dir / "restaurants.csv")
    df_proc_restaurants = pd.read_csv(proc_dir / "restaurants.csv")
    
    df_raw_riders = pd.read_csv(raw_dir / "riders.csv")
    df_proc_riders = pd.read_csv(proc_dir / "riders.csv")

    # Verify columns are identical (same names, same order)
    assert list(df_raw_orders.columns) == list(df_proc_orders.columns)
    assert list(df_raw_restaurants.columns) == list(df_proc_restaurants.columns)
    assert list(df_raw_riders.columns) == list(df_proc_riders.columns)

    # Verify no new columns were added
    assert df_raw_orders.shape[1] == df_proc_orders.shape[1]
    assert df_raw_restaurants.shape[1] == df_proc_restaurants.shape[1]
    assert df_raw_riders.shape[1] == df_proc_riders.shape[1]

    # Verify column names match exactly (NO renaming)
    raw_col_names = set(df_raw_orders.columns)
    proc_col_names = set(df_proc_orders.columns)
    assert raw_col_names == proc_col_names, f"Column names differ: {raw_col_names} vs {proc_col_names}"


def test_drop_actions_do_not_change_processed_schema(tmp_path: Path) -> None:
    """Column-drop actions should be ignored so the cleaned output preserves the raw schema."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    for path in (raw_dir, processed_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alpha", "Beta"],
            "lat": [12.97, 12.98],
            "lon": [77.59, 77.60],
        }
    )
    raw_df.to_csv(raw_dir / "restaurants.csv", index=False)

    manifest = {
        "manifest": [
            {
                "decision_id": "DEC-001",
                "dataset_id": "restaurants",
                "column": "name",
                "action": "DROP",
                "priority": "HIGH",
                "confidence": 0.95,
                "rationale": "Drop the name column.",
                "status": "APPROVED",
            }
        ]
    }
    (reports_dir / "approval_manifest.json").write_text(json.dumps(manifest))

    config = CleaningConfig(raw_dir=raw_dir, processed_dir=processed_dir, reports_dir=reports_dir)
    engine = CleaningEngine(config=config)

    engine.run(force_approve_all=True)

    proc_df = pd.read_csv(processed_dir / "restaurants.csv")
    assert list(proc_df.columns) == ["id", "name", "lat", "lon"]
    assert "name" in proc_df.columns


def test_manifest_action_validation_filters_unsupported_actions() -> None:
    """The cleaning engine should filter unsupported manifest actions before execution."""
    config = CleaningConfig()
    engine = CleaningEngine(config=config)
    decisions = [
        {
            "decision_id": "DEC-001",
            "dataset_id": "restaurants",
            "column": "avg_rating",
            "action": "STANDARDIZE",
            "priority": "LOW",
            "confidence": 0.50,
            "rationale": "Legacy scaling action.",
            "status": "APPROVED",
        },
        {
            "decision_id": "DEC-002",
            "dataset_id": "restaurants",
            "column": "avg_rating",
            "action": "IMPUTE",
            "priority": "HIGH",
            "confidence": 0.95,
            "rationale": "Fill missing ratings.",
            "status": "APPROVED",
        },
    ]

    filtered = engine._validate_manifest_actions(decisions)
    assert len(filtered) == 1
    assert filtered[0]["action"] == "IMPUTE"


def test_cleaning_engine_orchestration(mock_raw_and_manifest: tuple[Path, Path, Path]) -> None:
    """Integration test checking standard clean run and rollback restore logic."""
    raw_dir, proc_dir, rep_dir = mock_raw_and_manifest

    # Record checksum of raw files before cleaning to assert they are not modified
    checksums_before = {
        name: calculate_checksum(raw_dir / name)
        for name in ("restaurants.csv", "riders.csv", "orders.csv")
    }

    config = CleaningConfig(raw_dir=raw_dir, processed_dir=proc_dir, reports_dir=rep_dir)
    engine = CleaningEngine(config=config)

    # Run with standard manifest approval filtering (DEC-004 pending)
    paths = engine.run(force_approve_all=False)

    # Assert reports are generated
    assert paths["cleaning_summary_json"].exists()
    assert paths["cleaning_timeline_json"].exists()
    assert paths["before_after_quality_json"].exists()
    assert paths["cleaning_report_md"].exists()
    assert paths["rollback_manifest_json"].exists()

    # Assert processed datasets are created
    assert (proc_dir / "restaurants.csv").exists()
    assert (proc_dir / "riders.csv").exists()
    assert (proc_dir / "orders.csv").exists()

    # Verify duplicate rows removed in processed restaurants
    proc_rest = pd.read_csv(proc_dir / "restaurants.csv")
    assert len(proc_rest) == 2  # Originally 3, 1 duplicate removed

    # Verify the outlier is removed, while schema-changing DROP actions are ignored.
    # Originally: O001 (valid), O002 (orphan restaurant_id), O003 (negative delivery time)
    # After REMOVE_OUTLIERS: O001, O002 (removed O003)
    # DROP actions are filtered to preserve the raw schema and do not remove orphan rows.
    proc_ord = pd.read_csv(proc_dir / "orders.csv")
    assert len(proc_ord) == 2  # Originally 3, 1 negative outlier removed; orphan row retained

    # Verify pending approvals (DEC-004 coordinate flagging) not executed
    proc_rid = pd.read_csv(proc_dir / "riders.csv")
    assert len(proc_rid) == 2  # No rows removed since it was pending

    # Assert raw source files are completely unchanged
    for name, sum_before in checksums_before.items():
        assert calculate_checksum(raw_dir / name) == sum_before

    # Test Rollback Capability
    # Corrupt a processed file first
    proc_rest_file = proc_dir / "restaurants.csv"
    proc_rest_file.write_text("corrupted content")

    # Run rollback CLI command
    exit_code = main(["--reports-dir", str(rep_dir), "--rollback"])
    assert exit_code == 0

    # Verify restaurant file is restored (should not be corrupted anymore)
    restored_df = pd.read_csv(proc_rest_file)
    assert len(restored_df) == 3  # Still correct raw row count


def test_cli_force_approve_all(mock_raw_and_manifest: tuple[Path, Path, Path]) -> None:
    """Verify CLI --force-approve-all execution overrides status filter."""
    raw_dir, proc_dir, rep_dir = mock_raw_and_manifest
    
    # Run CLI with force approve flag (this should trigger DEC-004 and run GPS cleaner on riders)
    exit_code = main([
        "--raw-dir", str(raw_dir),
        "--processed-dir", str(proc_dir),
        "--reports-dir", str(rep_dir),
        "--force-approve-all"
    ])
    assert exit_code == 0

    # DEC-004 is latitude flagging. GPS cleaner should nullify invalid GPS (rider_id D002 has lat=999.0)
    proc_rid = pd.read_csv(proc_dir / "riders.csv")
    assert len(proc_rid) == 2  # Originally 2, no rows dropped
    assert pd.isna(proc_rid["latitude"].iloc[1])  # Lat=999.0 nullified
