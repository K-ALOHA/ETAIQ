"""New unit tests to verify specific cleaning operations and improvements in the Cleaning Execution Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.cleaning.cleaning_engine import CleaningEngine
from ml.cleaning.config import CleaningConfig
from ml.cleaning.duplicate_handler import DuplicateHandler
from ml.cleaning.gps_cleaner import GpsCleaner
from ml.cleaning.imputation import ImputationHandler
from ml.cleaning.integrity_cleaner import IntegrityCleaner
from ml.cleaning.timestamp_cleaner import TimestampCleaner


def test_duplicate_removal_actually_removes_rows() -> None:
    """Verify that DuplicateHandler drops exact duplicates from a DataFrame."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "value": ["A", "B", "B", "C"]
    })
    handler = DuplicateHandler()
    cleaned, result = handler.execute(df, "test")
    assert len(cleaned) == 3
    assert result.success
    assert result.details["dropped_count"] == 1


def test_timestamp_cleaner_standardizes_and_parses_dayfirst() -> None:
    """Verify that TimestampCleaner parses dates like '18/05/2026 22:56' and formats to standard ISO format."""
    df = pd.DataFrame({
        "timestamp": ["18/05/2026 22:56", "2026-05-19 23:00:00"]
    })
    handler = TimestampCleaner()
    cleaned, result = handler.execute(df, "test", "timestamp")
    assert cleaned["timestamp"].iloc[0] == "2026-05-18 22:56:00"
    assert cleaned["timestamp"].iloc[1] == "2026-05-19 23:00:00"
    assert result.success


def test_datatype_and_integrity_cleaner_converts_float_ids() -> None:
    """Verify that IntegrityCleaner and DatatypeCleaner correctly convert float IDs like 5764.0 to clean integer strings '5764'."""
    df = pd.DataFrame({
        "rider_id": [5764.0, 3377.0, None]
    })
    ref_keys = {"5764", "3377"}
    handler = IntegrityCleaner()
    cleaned, result = handler.execute(df, "test", "rider_id", "FLAG", reference_keys=ref_keys)
    assert cleaned["rider_id"].iloc[0] == "5764"
    assert cleaned["rider_id"].iloc[1] == "3377"
    assert pd.isna(cleaned["rider_id"].iloc[2])
    assert not cleaned["rider_id_is_orphan"].iloc[0]


def test_gps_cleaner_fixes_invalid_coordinates_when_flagged() -> None:
    """Verify that GpsCleaner nullifies out-of-bounds coordinates (sets to NaN) instead of dropping the row when action is FLAG."""
    df = pd.DataFrame({
        "latitude": [12.97, 999.0]
    })
    handler = GpsCleaner()
    # FLAG action should keep both rows but nullify the second coordinate
    cleaned, result = handler.execute(df, "test", "latitude", action="FLAG")
    assert len(cleaned) == 2
    assert cleaned["latitude"].iloc[0] == 12.97
    assert pd.isna(cleaned["latitude"].iloc[1])
    assert result.details["corrected_count"] == 1


def test_imputation_changes_null_values() -> None:
    """Verify that ImputationHandler fills null values using median for numeric and mode for categorical."""
    df = pd.DataFrame({
        "numeric": [10.0, 20.0, None],
        "categorical": ["Apple", "Apple", None]
    })
    handler = ImputationHandler()
    
    cleaned, result = handler.execute(df, "test", "numeric")
    assert cleaned["numeric"].iloc[2] == 15.0  # Median
    
    cleaned, result = handler.execute(cleaned, "test", "categorical")
    assert cleaned["categorical"].iloc[2] == "Apple"  # Mode


@pytest.fixture
def mock_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Setup mock raw, processed, and reports directories."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    
    raw_dir.mkdir()
    processed_dir.mkdir()
    reports_dir.mkdir()
    
    # Save a minimal raw file
    restaurants_df = pd.DataFrame({
        "id": [1, 2, 2],
        "name": ["A", "B", "B"],
        "lat": [12.97, 999.0, 999.0],  # Out of bound coordinate
        "lon": [77.59, 77.59, 77.59]
    })
    restaurants_df.to_csv(raw_dir / "restaurants.csv", index=False)
    
    # Manifest
    manifest = {
        "manifest": [
            {
                "decision_id": "DEC-001",
                "dataset_id": "restaurants",
                "column": None,
                "action": "REMOVE_DUPLICATES",
                "priority": "HIGH",
                "confidence": 1.0,
                "status": "APPROVED"
            },
            {
                "decision_id": "DEC-002",
                "dataset_id": "restaurants",
                "column": "lat",
                "action": "FLAG",
                "priority": "HIGH",
                "confidence": 1.0,
                "status": "APPROVED"
            }
        ]
    }
    (reports_dir / "approval_manifest.json").write_text(json.dumps(manifest))
    
    return raw_dir, processed_dir, reports_dir


def test_processed_dataframe_differs_from_raw_and_rollback_restores(mock_directories: tuple[Path, Path, Path]) -> None:
    """Verify that processed dataframe is modified, and rollback restores it."""
    raw_dir, proc_dir, rep_dir = mock_directories
    
    config = CleaningConfig(raw_dir=raw_dir, processed_dir=proc_dir, reports_dir=rep_dir)
    engine = CleaningEngine(config=config)
    
    # Execute cleaning
    _ = engine.run(force_approve_all=False)
    
    # Verify processed file was created and is different from raw
    raw_file = raw_dir / "restaurants.csv"
    proc_file = proc_dir / "restaurants.csv"
    
    assert proc_file.exists()
    
    raw_df = pd.read_csv(raw_file)
    proc_df = pd.read_csv(proc_file)
    
    # Raw had 3 rows; duplicate handler should reduce to 2 rows
    assert len(raw_df) == 3
    assert len(proc_df) == 2
    
    # Cleaning must preserve the original business schema and only repair values.
    assert list(raw_df.columns) == list(proc_df.columns)
    assert list(raw_df.columns) == ["id", "name", "lat", "lon"]
    # Check that lat=999.0 was set to NaN (nullified)
    assert pd.isna(proc_df["lat"].iloc[1])
    
    # Verify rollback
    # Corrupt the processed file
    proc_file.write_text("corrupted data")
    
    # Run rollback
    from ml.cleaning.rollback import RollbackManager
    manifest_name = config.rollback_manifest_filename
    rollback_mgr = RollbackManager(rep_dir / manifest_name)
    success = rollback_mgr.execute_rollback()
    assert success
    
    # Read restored file and verify it is not corrupted and has 3 rows (raw backup row count)
    restored_df = pd.read_csv(proc_file)
    assert len(restored_df) == 3
