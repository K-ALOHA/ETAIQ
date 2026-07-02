"""Configuration settings for the training pipeline module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "ml" / "artifacts" / "models"
DEFAULT_MODEL_REGISTRY_PATH = REPO_ROOT / "ml" / "data" / "training" / "model_registry.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training pipeline architecture."""

    project_root: Path = field(default_factory=lambda: REPO_ROOT)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    model_registry_path: Path = field(default_factory=lambda: DEFAULT_MODEL_REGISTRY_PATH)
    random_seed: int = 42
    log_level: str = "INFO"


DEFAULT_TRAINING_CONFIG = TrainingConfig()
