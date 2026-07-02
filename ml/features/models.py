"""Data models for the feature engineering module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FeatureMetadata:
    """Metadata describing a feature in the registry."""

    name: str
    source: str = "unknown"
    feature_type: str = "unknown"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureRegistry:
    """Container for registered features."""

    features: list[FeatureMetadata] = field(default_factory=list)
    version: str = "0.1.0"


@dataclass
class PipelineState:
    """Current state of the feature pipeline."""

    status: str = "READY"
    step: str = "initialized"
    message: str = "Pipeline initialized."


@dataclass
class FeatureSummary:
    """High-level summary of engineered feature output."""

    dataset_name: str = ""
    total_features: int = 0
    numeric_features: int = 0
    categorical_features: int = 0
    output_path: Optional[Path] = None
