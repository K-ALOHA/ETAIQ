"""Production training pipeline package for ETAIQ."""

from .config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from .logging_config import TrainingLogger
from .models import ModelDefinition

__all__ = [
    "DEFAULT_TRAINING_CONFIG",
    "TrainingConfig",
    "TrainingLogger",
    "ModelDefinition",
]

# Optional import that requires xgboost
try:
    from .registry import ModelRegistry
    __all__.append("ModelRegistry")
except ImportError:
    pass

