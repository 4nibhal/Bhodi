from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bhodi_platform.conversation.infrastructure import (
    build_retriever,
    build_vectorstore,
)
from bhodi_platform.conversation.settings import ConversationSettings
from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry
from bhodi_platform.retrieval.runtime import get_embeddings


EmbeddingsFactory = Callable[[], Any]


def initialize_persistent_runtime(
    persist_directory: str | None = None,
    *,
    settings: ConversationSettings | None = None,
    embeddings_factory: EmbeddingsFactory | None = None,
) -> Any:
    resolved_settings = settings or ConversationSettings.from_environment()
    if persist_directory is not None:
        resolved_settings = ConversationSettings(
            persist_directory=Path(persist_directory),
            collection_name=resolved_settings.collection_name,
            retriever_k=resolved_settings.retriever_k,
        )
    embeddings = (embeddings_factory or get_embeddings)()
    return build_vectorstore(resolved_settings, embeddings)


_persistent_runtime_registry = RuntimeRegistry(
    lambda: initialize_persistent_runtime(
        str(ConversationSettings.from_environment().persist_directory)
    )
)


def start_persistent_runtime() -> Any:
    return _persistent_runtime_registry.start()


def get_persistent_runtime() -> Any:
    return _persistent_runtime_registry.get()


def reset_persistent_runtime() -> None:
    _persistent_runtime_registry.reset()


def stop_persistent_runtime() -> None:
    _persistent_runtime_registry.stop()


def get_persistent_vectorstore() -> Any:
    return get_persistent_runtime()


def get_persistent_retriever(conversation_id: str | None = None) -> Any:
    settings = ConversationSettings.from_environment()
    return build_retriever(
        get_persistent_vectorstore(),
        settings,
        conversation_id=conversation_id,
    )
