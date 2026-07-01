"""Audit logger to maintain a full trace of cleaning actions executed."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ml.cleaning.logging_config import get_logger
from ml.cleaning.models import AuditRecord

logger = get_logger(__name__)


class AuditLogger:
    """Manages the cleaning execution audit trail."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def log_success(
        self,
        dataset_id: str,
        column: str | None,
        action: str,
        records_before: int,
        records_after: int,
        details: dict[str, Any],
    ) -> AuditRecord:
        """Create and store a success audit record."""
        record = AuditRecord(
            timestamp=datetime.now(UTC).isoformat(),
            dataset_id=dataset_id,
            column=column,
            action=action,
            records_before=records_before,
            records_after=records_after,
            details=details,
            status="SUCCESS",
        )
        self._records.append(record)
        return record

    def log_failure(
        self,
        dataset_id: str,
        column: str | None,
        action: str,
        records_before: int,
        error_message: str,
    ) -> AuditRecord:
        """Create and store a failure audit record."""
        record = AuditRecord(
            timestamp=datetime.now(UTC).isoformat(),
            dataset_id=dataset_id,
            column=column,
            action=action,
            records_before=records_before,
            records_after=records_before,
            details={},
            status="FAILED",
            error_message=error_message,
        )
        self._records.append(record)
        return record

    def get_records(self) -> list[AuditRecord]:
        """Return the complete audit trail."""
        return self._records
