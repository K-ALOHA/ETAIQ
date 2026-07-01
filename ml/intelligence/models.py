"""Shared data models for the Dataset Intelligence Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ColumnRole(str, Enum):
    """Heuristic classification roles for dataset columns."""

    IDENTIFIER = "identifier"
    FOREIGN_KEY = "foreign_key"
    TIMESTAMP = "timestamp"
    NUMERIC_FEATURE = "numeric_feature"
    CATEGORICAL_FEATURE = "categorical_feature"
    BOOLEAN_FEATURE = "boolean_feature"
    TEXT = "text"
    HIGH_CARDINALITY = "high_cardinality"
    POTENTIAL_TARGET = "potential_target"
    POTENTIAL_LEAKAGE = "potential_leakage"
    PII = "pii"
    DISTRACTOR = "distractor"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class FeatureRecommendation(str, Enum):
    """Feature usefulness recommendation for modeling."""

    USEFUL = "useful_feature"
    WEAK = "weak_feature"
    IGNORE = "ignore"
    DERIVED_CANDIDATE = "derived_feature_candidate"


@dataclass
class ScannedDataset:
    """Metadata for a discovered CSV file."""

    dataset_id: str
    filename: str
    relative_path: str
    absolute_path: str
    row_count: int
    column_count: int
    memory_bytes: int
    sampled: bool
    sample_rows: int


@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""

    name: str
    inferred_dtype: str
    pandas_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    duplicate_rows: int
    sample_values: list[Any]
    is_numeric: bool
    is_categorical: bool
    is_boolean: bool
    is_datetime: bool
    is_text: bool
    is_high_cardinality: bool
    roles: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Complete profile for one discovered dataset."""

    dataset_id: str
    filename: str
    relative_path: str
    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)
    column_groups: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Relationship:
    """Detected join relationship between two datasets."""

    source_dataset: str
    source_column: str
    target_dataset: str
    target_column: str
    relationship_type: str
    join_confidence: float
    overlap_ratio: float
    optional: bool
    required: bool
    cardinality: str
    join_type: str = "left"
    reason: str = ""
    referential_integrity: float = 0.0
    tier: str = "possible"
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    business_justification: str = ""
    merge_risk: str = ""
    sql_join_example: str = ""


@dataclass
class FeatureCandidate:
    """Feature engineering recommendation for a column."""

    dataset_id: str
    column: str
    classification: str
    recommendation: str
    confidence: float
    reason: str = ""
    encoding: str | None = None
    scaling: str | None = None
    engineering: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class TargetCandidate:
    """Ranked prediction target candidate."""

    dataset_id: str
    column: str
    rank: int
    score: float
    confidence: float
    tier: str
    target_type: str
    rationale: str
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class LeakageFinding:
    """Suspected target leakage in a column."""

    dataset_id: str
    column: str
    severity: str
    confidence: float
    rationale: str
    recommendation: str
    related_target: str | None = None


@dataclass
class IntelligenceReport:
    """Aggregated output of a full intelligence run."""

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    data_dir: str = ""
    reports_dir: str = ""
    datasets: list[DatasetProfile] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    feature_candidates: list[FeatureCandidate] = field(default_factory=list)
    target_candidates: list[TargetCandidate] = field(default_factory=list)
    leakage_findings: list[LeakageFinding] = field(default_factory=list)
    merge_strategies: list[dict[str, Any]] = field(default_factory=list)
    schema_registry: dict[str, Any] = field(default_factory=dict)
    version_report: dict[str, Any] = field(default_factory=dict)
    intelligence_score: dict[str, Any] = field(default_factory=dict)
