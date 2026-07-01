"""Data models for the Cleaning Execution Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditRecord:
    """Detailed log record for an individual cleaning action."""

    timestamp: str
    dataset_id: str
    column: str | None
    action: str
    records_before: int
    records_after: int
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"
    error_message: str | None = None


@dataclass
class ExecutorResult:
    """Output summary of executing a single cleaning action on a dataset."""

    success: bool
    message: str
    records_before: int
    records_after: int
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass
class TimelineEvent:
    """Execution step event for pipeline tracking."""

    step_name: str
    started_at: str
    finished_at: str
    duration_seconds: float
    status: str
    message: str


@dataclass
class RollbackEntry:
    """Rollback details for a single dataset."""

    dataset_id: str
    processed_file_path: str
    backup_file_path: str
    checksum_raw: str
    checksum_processed: str
    row_count: int


@dataclass
class CleaningSummary:
    """Summary of the full execution run."""

    started_at: str
    finished_at: str
    total_actions_attempted: int
    total_actions_successful: int
    total_actions_failed: int
    datasets_processed: list[str]
    timeline: list[TimelineEvent] = field(default_factory=list)
    audit_trail: list[AuditRecord] = field(default_factory=list)
