"""Pydantic models for the ETAIQ AI assistant."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    """Incoming assistant request payload."""

    message: str = Field(..., min_length=1, description="User message to process")
    conversation_id: str | None = Field(
        default=None, description="Existing conversation identifier"
    )
    context: dict[str, Any] | None = Field(
        default=None, description="Optional caller-supplied context"
    )


class ConversationTurn(BaseModel):
    """A single message turn stored for conversation continuity."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None


class AssistantResponse(BaseModel):
    """Structured assistant response returned to the caller."""

    response: str = Field(..., description="Plain-text assistant answer")
    sources: list[str] = Field(
        default_factory=list, description="Context sources used to compose the answer"
    )
    conversation_id: str = Field(
        ..., description="Conversation identifier for follow-up turns"
    )
