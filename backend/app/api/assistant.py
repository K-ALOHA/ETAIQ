"""Assistant API routes for the ETAIQ AI assistant."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from app.ai.assistant import ETAIQAssistantService
from app.ai.schemas import AssistantRequest, AssistantResponse
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@lru_cache(maxsize=1)
def get_assistant_service() -> ETAIQAssistantService:
    """Return a shared assistant service instance so conversation history persists across requests."""
    return ETAIQAssistantService()


def _is_llm_available(service: ETAIQAssistantService) -> bool:
    """Return whether the backing LLM client appears configured and ready."""
    client = getattr(service, "_llm_client", None)
    if client is None:
        return False

    api_key = getattr(client, "api_key", None)
    sdk_available = getattr(client, "_sdk_available", False)
    return bool(api_key) and bool(sdk_available)


@router.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Send a chat message to the ETAIQ AI assistant",
    description="Creates or continues a conversation and returns a plain-text assistant reply.",
)
async def chat_with_assistant(request: Request) -> AssistantResponse:
    """Handle an assistant chat request and return structured assistant output."""
    logger.info("assistant_chat_requested")

    try:
        payload = await request.json()
    except ValueError as exc:
        logger.warning("assistant_invalid_json", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid request") from exc

    if not isinstance(payload, dict):
        logger.warning("assistant_invalid_payload", payload_type=type(payload).__name__)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid request")

    message = payload.get("message")
    conversation_id = payload.get("conversation_id")

    if not isinstance(message, str) or not message.strip():
        logger.warning("assistant_invalid_message")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid request")

    if conversation_id is not None and not isinstance(conversation_id, str):
        logger.warning("assistant_invalid_conversation_id")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid request")

    service = get_assistant_service()
    if not _is_llm_available(service):
        logger.warning("assistant_llm_unavailable")

    try:
        assistant_request = AssistantRequest(
            message=message.strip(),
            conversation_id=conversation_id,
        )
        return await service.handle_message(assistant_request)
    except ValidationError as exc:
        logger.warning("assistant_invalid_request", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid request") from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("assistant_internal_error", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="assistant internal error") from exc
