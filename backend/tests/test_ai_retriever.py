from __future__ import annotations

from app.ai.assistant import ETAIQAssistantService
from app.ai.openrouter_client import OpenRouterClientError
from app.ai.retriever import ContextRetriever
from app.ai.schemas import AssistantRequest


class StubBuilder:
    def __init__(self, *, explainability_available: bool = True) -> None:
        self.calls: list[str] = []
        self.explainability_available = explainability_available

    def get_production_model_context(self) -> dict[str, object]:
        self.calls.append("production")
        return {"name": "XGBRegressor", "version": 2, "status": "Production"}

    def get_registry_context(self) -> dict[str, object]:
        self.calls.append("registry")
        return {
            "production_model": "XGBRegressor",
            "version": 2,
            "metrics": {"mae": 1.2, "rmse": 2.3, "r2": 0.81},
        }

    def get_monitoring_context(self) -> dict[str, object]:
        self.calls.append("monitoring")
        return {"monitoring_status": "healthy", "latency_ms": 120.0, "record_count": 42}

    def get_health_context(self) -> dict[str, object]:
        self.calls.append("health")
        return {"status": "healthy", "model_loaded": True}

    def get_dataset_context(self) -> dict[str, object]:
        self.calls.append("dataset")
        return {"record_count": 1000, "feature_names": ["lat", "lon"], "target_column": "eta"}

    def get_training_context(self) -> dict[str, object]:
        self.calls.append("training")
        return {
            "latest_training_runs": [
                {"model_name": "XGBRegressor", "version": 2, "training_timestamp": "2026-01-01"}
            ]
        }

    def get_latest_prediction_context(self) -> dict[str, object]:
        self.calls.append("prediction")
        return {
            "status": "available",
            "summary": {"prediction_count": 20, "model_name": "XGBRegressor"},
        }

    def get_eda_context(self) -> dict[str, object]:
        self.calls.append("eda")
        return {"status": "available", "summary": {"dataset_count": 1, "datasets": ["training"]}}

    def get_explainability_context(self) -> dict[str, object]:
        self.calls.append("explainability")
        if not self.explainability_available:
            return {"available": False, "summary": "No explainability information is available."}
        return {
            "available": True,
            "summary": "The model relied most on road network and traffic features.",
            "top_features": [
                {"feature_name": "lat", "importance": 0.42},
                {"feature_name": "lon", "importance": 0.24},
            ],
            "confidence": "moderate",
        }


class FailingLLMClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_text(self, prompt: str, *, timeout: int = 30, max_retries: int = 3) -> str:
        self.calls.append(prompt)
        raise OpenRouterClientError("OpenRouter unavailable")


def test_detect_intent_for_registry_queries() -> None:
    retriever = ContextRetriever(builder=StubBuilder())

    intents = retriever.detect_intent("What model is in production?")

    assert intents == ["registry"]


def test_retrieve_context_uses_only_relevant_sources() -> None:
    builder = StubBuilder()
    retriever = ContextRetriever(builder=builder)

    result = retriever.retrieve_context("How accurate is the current model?")

    assert result["context"]["registry_summary"]["production_model"] == "XGBRegressor"
    assert "dataset_summary" not in result["context"]
    assert result["sources"] == ["Model Registry"]
    assert builder.calls == ["registry", "production"]


def test_source_attribution_tracks_used_modules() -> None:
    retriever = ContextRetriever(builder=StubBuilder())

    result = retriever.retrieve_context("Explain today's monitoring.")

    assert "Monitoring" in result["sources"]
    assert "monitoring" in result["intents"]
    assert "explainability" in result["intents"]


def test_fallback_path_uses_retrieved_context() -> None:
    service = ETAIQAssistantService(
        client=FailingLLMClient(),
        context_builder=StubBuilder(),
        retriever=ContextRetriever(builder=StubBuilder()),
        conversation_manager=None,
    )

    response = service.handle_message(
        AssistantRequest(message="What model is in production?", conversation_id=None)
    )

    assert "XGBRegressor" in response.response
    assert response.sources == ["Model Registry"]


def test_prediction_explanation_retrieval_uses_explainability_context() -> None:
    builder = StubBuilder()
    retriever = ContextRetriever(builder=builder)

    result = retriever.retrieve_context("Explain this prediction.")

    assert result["context"]["explainability_summary"]["available"] is True
    assert result["sources"][-1] == "Explainability"


def test_missing_explainability_context_is_reported_gracefully() -> None:
    builder = StubBuilder(explainability_available=False)
    retriever = ContextRetriever(builder=builder)

    result = retriever.retrieve_context("Why was this ETA predicted?")

    assert result["context"]["explainability_summary"]["available"] is False
    assert "No explainability information" in result["context"]["explainability_summary"]["summary"]


def test_follow_up_conversation_keeps_explainability_context() -> None:
    class RecordingLLMClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_text(self, prompt: str, *, timeout: int = 30, max_retries: int = 3) -> str:
            self.prompts.append(prompt)
            return "Summary\nTop contributing features\nConfidence\nSuggested actions\nSources"

    client = RecordingLLMClient()
    service = ETAIQAssistantService(
        client=client,
        context_builder=StubBuilder(),
        retriever=ContextRetriever(builder=StubBuilder()),
        conversation_manager=None,
    )

    first = service.handle_message(
        AssistantRequest(message="Explain this prediction.", conversation_id="followup")
    )
    second = service.handle_message(AssistantRequest(message="Why?", conversation_id="followup"))

    assert first.sources[-1] == "Explainability"
    assert second.sources[-1] == "Explainability"
    assert len(client.prompts) == 2
