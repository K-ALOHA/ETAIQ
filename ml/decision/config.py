"""Configuration parameters for the Decision Intelligence Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = REPO_ROOT / "ml" / "reports"


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds and weights for decision analysis."""

    # File paths
    reports_dir: Path = field(default=DEFAULT_REPORTS_DIR)

    # Imputation vs Drop thresholds
    null_drop_threshold: float = 0.80  # Drop column if null rate > 80%
    null_impute_threshold: float = 0.05  # Impute if null rate > 5% (and <= 80%)

    # Outlier detection
    outlier_std_devs: float = 3.0  # Max std devs from mean to flag as outlier

    # Scoring weights for estimated quality improvement
    weight_completeness: float = 0.25
    weight_consistency: float = 0.25
    weight_integrity: float = 0.25
    weight_model_reliability: float = 0.25


DEFAULT_DECISION_CONFIG = DecisionConfig()
