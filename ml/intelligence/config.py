"""Central configuration for the Dataset Intelligence Engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Datetime detection
# ---------------------------------------------------------------------------

DATETIME_NAME_PATTERN = re.compile(
    r"(time|date|timestamp|datetime|created|updated|modified|at$|_at$|_on$|_ts$)",
    re.IGNORECASE,
)

DATETIME_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),  # ISO-8601
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # YYYY-MM-DD
    re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),  # YYYY-MM-DD HH:MM:SS
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),  # DD/MM/YYYY or MM/DD/YYYY
    re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$"),  # DD/MM/YYYY HH:MM
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
)

# ---------------------------------------------------------------------------
# Column classification patterns
# ---------------------------------------------------------------------------

ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|key|code|number|num)($|_)", re.IGNORECASE)
TIME_NAME_PATTERN = DATETIME_NAME_PATTERN
PII_NAME_PATTERN = re.compile(
    r"(email|phone|mobile|ssn|address|first_name|last_name|full_name|name$)",
    re.IGNORECASE,
)
TARGET_NAME_PATTERN = re.compile(
    r"(target|label|duration|minutes|hours|seconds|delay|eta|outcome|score|amount|total|time)",
    re.IGNORECASE,
)
POST_EVENT_TARGET_PATTERN = re.compile(
    r"(actual|completed|delivered|observed|real|final|delivery_time|elapsed|duration)",
    re.IGNORECASE,
)
PRE_EVENT_FEATURE_PATTERN = re.compile(
    r"(promised|planned|estimated|estimate|expected|scheduled|forecast|projected)",
    re.IGNORECASE,
)
COORDINATE_NAME_PATTERN = re.compile(
    r"(^|_)(lat|lon|lng|latitude|longitude)($|_)",
    re.IGNORECASE,
)
METADATA_NAME_PATTERN = re.compile(
    r"(created|updated|modified|version|source|ingested|batch|partition)",
    re.IGNORECASE,
)
POST_EVENT_LEAKAGE_PATTERN = re.compile(
    r"(delivered|delivery|completed|finished|actual|outcome|result)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntelligenceConfig:
    """Thresholds and sampling parameters for intelligence inference."""

    # Sampling / performance
    sample_rows: int = 10_000
    chunk_size: int = 100_000
    datetime_sample_size: int = 50
    datetime_match_ratio: float = 0.8

    # Profiling
    high_cardinality_ratio: float = 0.5
    text_avg_length: int = 40
    identifier_uniqueness_ratio: float = 0.95

    # Target detection
    target_strong_threshold: float = 0.85
    target_possible_threshold: float = 0.50
    target_weak_threshold: float = 0.14
    target_post_event_boost: float = 0.35
    target_pre_event_penalty: float = 0.40
    target_coordinate_penalty: float = 0.50
    target_numeric_boost: float = 0.25
    target_name_boost: float = 0.20

    # Feature recommendation
    feature_required_threshold: float = 0.85
    feature_recommended_threshold: float = 0.65
    feature_optional_threshold: float = 0.40
    feature_high_null_penalty: float = 0.80

    # Relationships
    overlap_threshold: float = 0.60
    overlap_high_threshold: float = 0.85
    name_boost: float = 0.15
    uniqueness_weight: float = 0.20
    referential_integrity_weight: float = 0.30
    min_relationship_confidence: float = 0.55

    # Leakage
    leakage_correlation_high: float = 0.98
    leakage_correlation_medium: float = 0.90
    leakage_correlation_low: float = 0.75
    leakage_min_correlation_samples: int = 3

    # Intelligence score weights
    score_discovery_weight: float = 0.20
    score_relationship_weight: float = 0.20
    score_schema_weight: float = 0.20
    score_feature_weight: float = 0.20
    score_target_weight: float = 0.20

    # Regex patterns (shared references)
    datetime_name_pattern: re.Pattern[str] = field(
        default=DATETIME_NAME_PATTERN, repr=False
    )
    datetime_value_patterns: tuple[re.Pattern[str], ...] = field(
        default=DATETIME_VALUE_PATTERNS,
        repr=False,
    )


DEFAULT_CONFIG = IntelligenceConfig()
