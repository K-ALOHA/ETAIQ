"""Intent-aware context retrieval for the ETAIQ AI assistant."""

from __future__ import annotations

import re
import time
from typing import Any

from app.ai.context import ContextBuilder
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextRetriever:
    """Retrieve only the ETAIQ context relevant to a user question."""

    def __init__(self, *, builder: ContextBuilder | None = None, cache_ttl_seconds: float = 30.0) -> None:
        self.builder = builder or ContextBuilder()
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_fetched_at: dict[str, float] = {}

    def retrieve_context(self, question: str) -> dict[str, Any]:
        """Detect the intent for the question and retrieve only the relevant context."""
        intents = self.detect_intent(question)
        cache_key = "|".join(intents) if intents else "default"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None and cache_key in self._cache_fetched_at and now - self._cache_fetched_at[cache_key] < self.cache_ttl_seconds:
            return {"intents": intents, "context": cached["context"], "sources": cached["sources"]}

        context: dict[str, Any] = {}
        sources: list[str] = []

        if "registry" in intents:
            registry_context = self.builder.get_registry_context()
            production_context = self.builder.get_production_model_context()
            context["registry_summary"] = registry_context
            context["production_model"] = production_context
            sources.append("Model Registry")

        if "monitoring" in intents:
            monitoring_context = self.builder.get_monitoring_context()
            context["monitoring_summary"] = monitoring_context
            sources.append("Monitoring")

        if "prediction" in intents:
            prediction_context = self.builder.get_latest_prediction_context()
            context["latest_prediction"] = prediction_context
            sources.append("Prediction Metadata")

        if "explainability" in intents or "prediction" in intents:
            explainability_context = self._get_explainability_context()
            context["explainability_summary"] = explainability_context
            sources.append("Explainability")

        if "health" in intents:
            health_context = self.builder.get_health_context()
            context["health_context"] = health_context
            sources.append("Health Status")

        if "dataset" in intents:
            dataset_context = self.builder.get_dataset_context()
            context["dataset_summary"] = dataset_context
            sources.append("Dataset Metadata")

        if "training" in intents:
            training_context = self.builder.get_training_context()
            context["training_summary"] = training_context
            sources.append("Training Metadata")

        if "drift" in intents:
            eda_context = self.builder.get_eda_context()
            context["eda_summary"] = eda_context
            sources.append("Drift Detection")

        if not context:
            context["health_context"] = self.builder.get_health_context()
            sources.append("Health Status")

        response = {"intents": intents, "context": context, "sources": sources}
        self._cache[cache_key] = response
        self._cache_fetched_at[cache_key] = now
        return response

    def detect_intent(self, question: str) -> list[str]:
        """Return a lightweight intent list derived from keywords in the user question."""
        lowered = question.lower()
        intents: list[str] = []

        if any(keyword in lowered for keyword in ["model", "production", "registry", "version", "accuracy", "compare", "summary"]):
            intents.append("registry")

        if any(keyword in lowered for keyword in ["monitor", "monitoring", "latency", "metric", "metrics", "today", "status"]):
            intents.append("monitoring")

        if any(keyword in lowered for keyword in ["prediction", "predict", "explain this prediction", "latest prediction"]):
            intents.append("prediction")

        if any(keyword in lowered for keyword in ["health", "system", "status"]):
            intents.append("health")

        if any(keyword in lowered for keyword in ["dataset", "feature", "features", "data", "target", "record"]):
            intents.append("dataset")

        if any(keyword in lowered for keyword in ["training", "train", "trained", "training run", "latest training"]):
            intents.append("training")

        if any(keyword in lowered for keyword in ["explain", "why", "important", "feature affect", "affect eta", "feature importance", "contributed most", "confidence", "confidence low", "what increased", "what decreased"]):
            intents.append("explainability")

        if any(keyword in lowered for keyword in ["drift", "shift", "deviation"]):
            intents.append("drift")

        return intents or ["health"]

    def _get_explainability_context(self) -> dict[str, Any]:
        """Return explainability context from the builder when available."""
        if hasattr(self.builder, "get_explainability_context"):
            return self.builder.get_explainability_context()
        return {"available": False, "summary": "No explainability information is available."}
