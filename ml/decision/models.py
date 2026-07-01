"""Shared data models for the Decision Intelligence Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionVerb(str, Enum):
    """Supported deterministic cleaning actions."""

    KEEP = "KEEP"
    DROP = "DROP"
    IMPUTE = "IMPUTE"
    FLAG = "FLAG"
    REMOVE_DUPLICATES = "REMOVE_DUPLICATES"
    REMOVE_OUTLIERS = "REMOVE_OUTLIERS"
    FIX_DATATYPE = "FIX_DATATYPE"
    LEAVE_UNCHANGED = "LEAVE_UNCHANGED"
    STANDARDIZE_TIMESTAMP = "STANDARDIZE_TIMESTAMP"
    FIX_GPS = "FIX_GPS"
    REPAIR_FOREIGN_KEY = "REPAIR_FOREIGN_KEY"
    TRIM_WHITESPACE = "TRIM_WHITESPACE"
    FORMAT_STRING = "FORMAT_STRING"


class PriorityLevel(str, Enum):
    """Priority levels for cleaning recommendations."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class ActionableIssue:
    """An actionable issue detected in previous intelligence outputs."""

    issue_type: str
    dataset_id: str
    column: str | None
    description: str
    severity: PriorityLevel
    source_report: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningRecommendation:
    """A deterministic cleaning recommendation generated from an actionable issue."""

    action: ActionVerb
    dataset_id: str
    column: str | None
    confidence: float
    priority: PriorityLevel
    rationale: str
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactMetric:
    """Estimated improvement details for a single quality aspect."""

    baseline: float
    target: float
    delta: float


@dataclass
class ExpectedImpacts:
    """Aggregated estimated improvements across different data dimensions."""

    data_quality: ImpactMetric
    completeness: ImpactMetric
    consistency: ImpactMetric
    integrity: ImpactMetric
    model_reliability: ImpactMetric


@dataclass
class ApprovalManifestEntry:
    """A single clean action entry for the approval manifest."""

    decision_id: str
    dataset_id: str
    column: str | None
    action: ActionVerb
    priority: PriorityLevel
    confidence: float
    rationale: str
    status: str = "PENDING_APPROVAL"


@dataclass
class DecisionSummaryPayload:
    """The complete decision output details for serialization."""

    generated_at: str
    total_issues_found: int
    total_recommendations: int
    overall_confidence: float
    issues_by_severity: dict[str, int]
    recommendations_by_action: dict[str, int]
    dataset_level_summaries: dict[str, Any]
