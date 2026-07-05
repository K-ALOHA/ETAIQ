"""Conversation memory and state management for the ETAIQ AI assistant."""

from __future__ import annotations

from enum import Enum
from typing import Any

from app.ai.memory import ConversationMemory


class ConversationState(str, Enum):
    NORMAL_CHAT = "normal_chat"
    PREDICTION = "prediction"
    WAITING_FIELD = "waiting_field"
    PREDICTION_COMPLETE = "prediction_complete"
    CANCELLED = "cancelled"


class ConversationManager:
    """Maintain recent conversation turns and per-conversation state in memory.

    History storage is delegated to :class:`ConversationMemory` so that the
    trimming logic and the public memory interface live in one place.
    """

    def __init__(self, history_limit: int = 20) -> None:
        self._memory = ConversationMemory(max_messages=history_limit)
        self._states: dict[str, ConversationState] = {}
        self._prediction_data: dict[str, dict[str, Any]] = {}

    def create_conversation_id(self) -> str:
        return ConversationMemory.new_session_id()

    def add_turn(self, conversation_id: str, *, role: str, content: str) -> None:
        if role == "user":
            self._memory.add_user_message(conversation_id, content)
        else:
            self._memory.add_assistant_message(conversation_id, content)

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self._memory.get_history(conversation_id)

    def get_state(self, conversation_id: str) -> ConversationState:
        return self._states.get(conversation_id, ConversationState.NORMAL_CHAT)

    def set_state(self, conversation_id: str, state: ConversationState) -> None:
        self._states[conversation_id] = state

    def get_prediction_data(self, conversation_id: str) -> dict[str, Any]:
        return self._prediction_data.get(conversation_id, {})

    def set_prediction_data(self, conversation_id: str, data: dict[str, Any]) -> None:
        self._prediction_data[conversation_id] = data

    def clear_prediction_data(self, conversation_id: str) -> None:
        self._prediction_data.pop(conversation_id, None)
        self._states.pop(conversation_id, None)

    def has_prediction_data(self, conversation_id: str) -> bool:
        return conversation_id in self._prediction_data
