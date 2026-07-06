from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai.schemas import AssistantRequest, AssistantResponse
from app.api.assistant import router as assistant_router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant_router, prefix="/api/v1")
    return TestClient(app)


def test_assistant_chat_successful_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubService:
        def __init__(self) -> None:
            self._llm_client = SimpleNamespace(_sdk_available=True, api_key="test")

        async def handle_message(self, request: AssistantRequest) -> AssistantResponse:
            return AssistantResponse(
                response="Hello from ETAIQ",
                sources=["registry_summary"],
                conversation_id=request.conversation_id or "conv-123",
            )

    monkeypatch.setattr(
        "app.api.assistant.ETAIQAssistantService", lambda *args, **kwargs: StubService()
    )

    response = client.post(
        "/api/v1/assistant/chat",
        json={"message": "How is the model doing?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "Hello from ETAIQ"
    assert payload["conversation_id"] == "conv-123"
    assert payload["sources"] == ["registry_summary"]


def test_assistant_chat_invalid_payload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/assistant/chat",
        json={"message": 42},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid request"


def test_assistant_chat_llm_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubService:
        def __init__(self) -> None:
            self._llm_client = SimpleNamespace(_sdk_available=False, api_key="")

        async def handle_message(self, request: AssistantRequest) -> AssistantResponse:
            return AssistantResponse(
                response="Fallback response",
                sources=["Model Registry"],
                conversation_id=request.conversation_id or "conv-123",
            )

    monkeypatch.setattr("app.api.assistant.get_assistant_service", lambda: StubService())

    response = client.post(
        "/api/v1/assistant/chat",
        json={"message": "Tell me about the registry"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Fallback response"


def test_assistant_chat_conversation_continuation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str | None] = []

    class StubService:
        def __init__(self) -> None:
            self._llm_client = SimpleNamespace(_sdk_available=True, api_key="test")

        async def handle_message(self, request: AssistantRequest) -> AssistantResponse:
            captured.append(request.conversation_id)
            return AssistantResponse(
                response="continuing",
                sources=["monitoring_summary"],
                conversation_id=request.conversation_id or "new-conversation",
            )

    monkeypatch.setattr("app.api.assistant.get_assistant_service", lambda: StubService())

    first_response = client.post(
        "/api/v1/assistant/chat",
        json={"message": "Start a conversation", "conversation_id": "session-1"},
    )
    second_response = client.post(
        "/api/v1/assistant/chat",
        json={"message": "Continue it", "conversation_id": "session-1"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert captured == ["session-1", "session-1"]
