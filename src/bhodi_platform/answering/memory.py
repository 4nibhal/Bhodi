from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bhodi_platform.answering.ports import VectorStorePort
from bhodi_platform.ports.answering import ConversationMemoryPort


class DualVectorStoreConversationMemory(ConversationMemoryPort):
    def __init__(
        self,
        volatile_vectorstore_factory: Callable[[], VectorStorePort | Any],
        persistent_vectorstore_factory: Callable[[], VectorStorePort | Any],
    ) -> None:
        self._volatile_vectorstore_factory = volatile_vectorstore_factory
        self._persistent_vectorstore_factory = persistent_vectorstore_factory

    def append_turn(
        self,
        user_message: str,
        answer_text: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        texts = [user_message, answer_text]
        self._volatile_vectorstore_factory().add_texts(texts)
        self._persistent_vectorstore_factory().add_texts(
            texts,
            metadatas=[
                _conversation_metadata(conversation_id, role="user"),
                _conversation_metadata(conversation_id, role="assistant"),
            ],
        )


def _conversation_metadata(conversation_id: str | None, *, role: str) -> dict[str, str]:
    metadata = {"origin": "conversation", "role": role}
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id
    return metadata


def store_conversation_turn(
    user_message: str,
    answer_text: str,
    volatile_vectorstore: VectorStorePort,
    persistent_vectorstore: VectorStorePort,
    *,
    conversation_id: str | None = None,
) -> None:
    DualVectorStoreConversationMemory(
        volatile_vectorstore_factory=lambda: volatile_vectorstore,
        persistent_vectorstore_factory=lambda: persistent_vectorstore,
    ).append_turn(
        user_message,
        answer_text,
        conversation_id=conversation_id,
    )
