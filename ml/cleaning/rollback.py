"""Rollback utility to manage backup catalogs and verify dataset integrity."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import RollbackEntry

logger = get_logger(__name__)


def calculate_checksum(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file.

    Returns empty string if file does not exist.
    """
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:
        logger.error("checksum_calculation_failed", path=str(path), error=str(exc))
        return ""


class RollbackManager:
    """Manages the creation and application of rollback manifests."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize with path to save manifest file."""
        self._manifest_path = manifest_path
        self._entries: list[RollbackEntry] = []

    def register_dataset(
        self,
        dataset_id: str,
        raw_path: Path,
        processed_path: Path,
        row_count: int,
    ) -> RollbackEntry:
        """Register a cleaned dataset and compute checksums."""
        checksum_raw = calculate_checksum(raw_path)
        checksum_proc = calculate_checksum(processed_path)

        entry = RollbackEntry(
            dataset_id=dataset_id,
            processed_file_path=str(processed_path.resolve()),
            backup_file_path=str(raw_path.resolve()),
            checksum_raw=checksum_raw,
            checksum_processed=checksum_proc,
            row_count=row_count,
        )
        self._entries.append(entry)
        return entry

    def save_manifest(self) -> Path:
        """Save registered entries as rollback_manifest.json."""
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "manifest": [
                {
                    "dataset_id": entry.dataset_id,
                    "processed_file_path": entry.processed_file_path,
                    "backup_file_path": entry.backup_file_path,
                    "checksum_raw": entry.checksum_raw,
                    "checksum_processed": entry.checksum_processed,
                    "row_count": entry.row_count,
                }
                for entry in self._entries
            ],
            "rollback_instructions": (
                "To rollback all processed datasets, copy backup_file_path to processed_file_path, "
                "or run python -m ml.cleaning.main --rollback"
            ),
        }
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("rollback_manifest_saved", path=str(self._manifest_path))
        return self._manifest_path

    def execute_rollback(self) -> bool:
        """Perform a restore operation based on the manifest file.

        Overwrites processed files with raw source files.
        """
        if not self._manifest_path.exists():
            logger.error("rollback_failed_no_manifest", path=str(self._manifest_path))
            return False

        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            manifest = payload.get("manifest", [])

            success = True
            for entry in manifest:
                raw_path = Path(entry["backup_file_path"])
                proc_path = Path(entry["processed_file_path"])

                if not raw_path.exists():
                    logger.error("rollback_source_missing", path=str(raw_path))
                    success = False
                    continue

                # Verify checksum_raw is correct
                curr_checksum = calculate_checksum(raw_path)
                if curr_checksum != entry["checksum_raw"]:
                    logger.warning("rollback_raw_checksum_mismatch", path=str(raw_path))

                # Copy raw back to processed (effectively rolling back)
                proc_path.write_bytes(raw_path.read_bytes())
                logger.info("rollback_file_restored", src=str(raw_path), dest=str(proc_path))

            return success
        except Exception as exc:
            logger.exception("rollback_execution_failed", error=str(exc))
            return False
