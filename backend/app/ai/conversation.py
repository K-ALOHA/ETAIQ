"""Conversation memory and state management for the ETAIQ AI assistant."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any
from uuid import uuid4


class ConversationState(str, Enum):
    NORMAL_CHAT = "normal_chat"
    PREDICTION = "prediction"
    WAITING_FIELD = "waiting_field"
    PREDICTION_COMPLETE = "prediction_complete"
    CANCELLED = "cancelled"


class ConversationManager:
    """Maintain recent conversation turns and per-conversation state in memory."""

    def __init__(self, history_limit: int = 8) -> None:
        self.history_limit = history_limit
        self._conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._states: dict[str, ConversationState] = {}
        self._prediction_data: dict[str, dict[str, Any]] = {}

    def create_conversation_id(self) -> str:
        return str(uuid4())

    def add_turn(self, conversation_id: str, *, role: str, content: str) -> None:
        history = self._conversations[conversation_id]
        history.append({"role": role, "content": content})
        if len(history) > self.history_limit:
            del history[: len(history) - self.history_limit]

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in self._conversations.get(conversation_id, [])
        ]

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
