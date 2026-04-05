"""
Volatile (in-memory) conversation memory adapter.

Stores conversation history in memory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bhodi_platform.application.config import ConversationConfig
    from bhodi_platform.domain.entities import ConversationTurn
    from bhodi_platform.domain.value_objects import ConversationId


class VolatileConversationMemoryAdapter:
    """
    In-memory conversation memory.

    Stores history in a dict. Lost on process restart.
    """

    def __init__(self, config: ConversationConfig) -> None:
        self._config = config
        self._max_history = config.max_history
        self._history: dict[str, list[ConversationTurn]] = defaultdict(list)

    async def add(
        self,
        conversation_id: ConversationId,
        turn: ConversationTurn,
    ) -> None:
        """Add a turn to conversation history."""
        key = str(conversation_id)
        self._history[key].append(turn)

        # Trim if max_history is set
        if self._max_history is not None:
            if len(self._history[key]) > self._max_history:
                self._history[key] = self._history[key][-self._max_history :]

    async def get_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation history."""
        key = str(conversation_id)
        history = self._history.get(key, [])

        if limit is not None:
            return history[-limit:]
        return list(history)

    async def clear(self, conversation_id: ConversationId) -> None:
        """Clear conversation history."""
        key = str(conversation_id)
        if key in self._history:
            del self._history[key]
