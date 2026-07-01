"""Shared data models for the validation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation step on one dataset.

    Attributes:
        validator_name: Name of the validator that produced this result.
        dataset_name: Logical dataset identifier (e.g. ``orders``).
        passed: Whether the validation met quality thresholds.
        score: Component score between 0 and 100.
        details: Structured findings for reporting.
        duration_seconds: Wall-clock execution time.
    """

    validator_name: str
    dataset_name: str
    passed: bool
    score: float
    details: dict[str, Any]
    duration_seconds: float


@dataclass
class ValidationSummary:
    """Aggregated validation run across all datasets.

    Attributes:
        results: Individual validation results.
        quality_score: Weighted overall score between 0 and 100.
        component_scores: Per-category weighted contributions.
        started_at: UTC timestamp when validation began.
        finished_at: UTC timestamp when validation completed.
        data_dir: Directory containing input CSV files.
        reports_dir: Directory where reports were written.
    """

    results: list[ValidationResult] = field(default_factory=list)
    quality_score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    data_dir: str = ""
    reports_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary to a JSON-compatible dictionary.

        Returns:
            dict[str, Any]: Summary payload for report generation.
        """
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "data_dir": self.data_dir,
            "reports_dir": self.reports_dir,
            "quality_score": round(self.quality_score, 2),
            "component_scores": {
                k: round(v, 2) for k, v in self.component_scores.items()
            },
            "results": [
                {
                    "validator_name": result.validator_name,
                    "dataset_name": result.dataset_name,
                    "passed": result.passed,
                    "score": round(result.score, 2),
                    "duration_seconds": round(result.duration_seconds, 4),
                    "details": result.details,
                }
                for result in self.results
            ],
        }
