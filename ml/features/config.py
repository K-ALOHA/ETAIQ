"""Configuration settings for the feature engineering module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "ml" / "data" / "processed"
DEFAULT_OUTPUT_FEATURE_DIR = REPO_ROOT / "ml" / "features" / "output"
DEFAULT_FEATURE_REGISTRY_PATH = REPO_ROOT / "ml" / "features" / "feature_registry.json"


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for the feature engineering architecture."""

    project_root: Path = field(default_factory=lambda: REPO_ROOT)
    processed_data_dir: Path = field(default_factory=lambda: DEFAULT_PROCESSED_DIR)
    output_feature_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_FEATURE_DIR)
    random_seed: int = 42
    default_scaler_name: str = "standard_scaler"
    default_encoder_name: str = "one_hot_encoder"
    log_level: str = "INFO"
    feature_registry_output_path: Path = field(
        default_factory=lambda: DEFAULT_FEATURE_REGISTRY_PATH
    )


DEFAULT_FEATURE_ENGINEERING_CONFIG = FeatureEngineeringConfig()
