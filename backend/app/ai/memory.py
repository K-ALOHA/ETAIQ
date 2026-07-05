"""Conversation memory for the ETAIQ AI assistant.

Stores per-session message history in memory (no persistence across restarts).
Trims to the most recent `max_messages` turns automatically.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import uuid4


class ConversationMemory:
    """Lightweight in-memory conversation history store.

    Each session is identified by a UUID string.  History is a list of
    ``{"role": ..., "content": ...}`` dicts ready to be forwarded to OpenRouter.
    Older messages are dropped automatically once the window exceeds
    ``max_messages``.
    """

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.max_messages = max_messages
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def new_session_id() -> str:
        """Return a fresh UUID string suitable for use as a session key."""
        return str(uuid4())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of the message history for *session_id*."""
        return list(self._store[session_id])

    def add_user_message(self, session_id: str, message: str) -> None:
        """Append a user turn and trim if necessary."""
        self._append(session_id, "user", message)

    def add_assistant_message(self, session_id: str, message: str) -> None:
        """Append an assistant turn and trim if necessary."""
        self._append(session_id, "assistant", message)

    def clear_history(self, session_id: str) -> None:
        """Remove all stored turns for *session_id*."""
        self._store.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, session_id: str, role: str, content: str) -> None:
        history = self._store[session_id]
        history.append({"role": role, "content": content})
        if len(history) > self.max_messages:
            del history[: len(history) - self.max_messages]
