"""Data models for the training pipeline module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    """Metadata describing a model registered for training."""

    model_name: str
    task: str = "regression"
    needs_scaling: bool = False
    supports_feature_importance: bool = False
    default_parameters: dict[str, Any] = field(default_factory=dict)
