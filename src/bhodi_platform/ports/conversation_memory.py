"""
Conversation memory port definition.

Defines the contract for conversation memory adapters.
"""

from __future__ import annotations

from typing import Protocol

from bhodi_platform.domain.entities import ConversationTurn
from bhodi_platform.domain.value_objects import ConversationId


class ConversationMemoryPort(Protocol):
    """
    Protocol for conversation memory storage.

    Adapters implementing this port handle persisting
    and retrieving conversation history.
    """

    async def add(
        self,
        conversation_id: ConversationId,
        turn: ConversationTurn,
    ) -> None:
        """
        Add a conversation turn to history.

        Args:
            conversation_id: The conversation's unique ID.
            turn: The conversation turn to add.

        Raises:
            ConversationError: If storage fails.
        """
        ...

    async def get_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """
        Get conversation history.

        Args:
            conversation_id: The conversation's unique ID.
            limit: Maximum number of turns to return (most recent).

        Returns:
            List of conversation turns.

        Raises:
            ConversationError: If retrieval fails.
        """
        ...

    async def clear(self, conversation_id: ConversationId) -> None:
        """
        Clear conversation history.

        Args:
            conversation_id: The conversation's unique ID.

        Raises:
            ConversationError: If clear fails.
        """
        ...
