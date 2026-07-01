"""Recursive CSV discovery and loading for the intelligence engine."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import ScannedDataset

logger = get_logger(__name__)

DEFAULT_SAMPLE_ROWS = 10_000
CHUNK_SIZE = 100_000


def _slugify(value: str) -> str:
    """Convert a path fragment into a stable dataset identifier.

    Args:
        value: Raw string such as a filename stem or relative path.

    Returns:
        str: Lowercase slug safe for use as a dictionary key.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "dataset"


def dataset_id_from_path(csv_path: Path, root_dir: Path) -> str:
    """Derive a unique dataset identifier from a CSV path.

    Args:
        csv_path: Absolute or relative path to a CSV file.
        root_dir: Root directory being scanned.

    Returns:
        str: Stable dataset identifier.
    """
    relative = csv_path.relative_to(root_dir)
    parts = list(relative.parts)
    if parts:
        parts[-1] = Path(parts[-1]).stem
    return _slugify("_".join(parts))


def discover_csv_files(data_dir: Path) -> list[Path]:
    """Recursively discover all CSV files under a directory.

    Args:
        data_dir: Root directory to scan.

    Returns:
        list[Path]: Sorted list of discovered CSV paths.
    """
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.csv"))


def count_csv_rows(path: Path) -> int:
    """Count rows in a CSV without loading the full file.

    Args:
        path: CSV file path.

    Returns:
        int: Number of data rows.
    """
    total = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        total += len(chunk)
    return total


def load_dataset_frame(
    path: Path, sample_rows: int = DEFAULT_SAMPLE_ROWS
) -> tuple[pd.DataFrame, bool]:
    """Load a CSV, sampling when the file exceeds the sample threshold.

    Args:
        path: CSV file path.
        sample_rows: Maximum rows to load for profiling large files.

    Returns:
        tuple[pd.DataFrame, bool]: DataFrame and whether sampling was applied.
    """
    row_count = count_csv_rows(path)
    if row_count <= sample_rows:
        return pd.read_csv(path, low_memory=False), False
    return pd.read_csv(path, nrows=sample_rows, low_memory=False), True


class DatasetScanner:
    """Discovers and loads CSV datasets from a raw data directory."""

    def __init__(self, data_dir: Path, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> None:
        """Initialize the scanner.

        Args:
            data_dir: Root directory to scan recursively.
            sample_rows: Row cap for large-file profiling.
        """
        self._data_dir = data_dir
        self._sample_rows = sample_rows

    def scan(self) -> tuple[list[ScannedDataset], dict[str, pd.DataFrame]]:
        """Discover and load all CSV datasets.

        Returns:
            tuple[list[ScannedDataset], dict[str, pd.DataFrame]]: Scan metadata and frames.
        """
        logger.info("dataset_scan_start", data_dir=str(self._data_dir))
        start = time.perf_counter()

        csv_files = discover_csv_files(self._data_dir)
        metadata: list[ScannedDataset] = []
        frames: dict[str, pd.DataFrame] = {}

        for csv_path in csv_files:
            dataset_id = dataset_id_from_path(csv_path, self._data_dir)
            logger.info("dataset_discovered", dataset_id=dataset_id, path=str(csv_path))

            load_start = time.perf_counter()
            frame, sampled = load_dataset_frame(csv_path, self._sample_rows)
            row_count = count_csv_rows(csv_path) if sampled else len(frame)
            memory_bytes = int(frame.memory_usage(deep=True).sum())

            scanned = ScannedDataset(
                dataset_id=dataset_id,
                filename=csv_path.name,
                relative_path=str(csv_path.relative_to(self._data_dir)),
                absolute_path=str(csv_path.resolve()),
                row_count=row_count,
                column_count=len(frame.columns),
                memory_bytes=memory_bytes,
                sampled=sampled,
                sample_rows=len(frame),
            )
            metadata.append(scanned)
            frames[dataset_id] = frame

            logger.info(
                "dataset_load_end",
                dataset_id=dataset_id,
                rows=row_count,
                columns=len(frame.columns),
                sampled=sampled,
                duration_seconds=round(time.perf_counter() - load_start, 4),
            )

        logger.info(
            "dataset_scan_end",
            datasets_found=len(metadata),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return metadata, frames
