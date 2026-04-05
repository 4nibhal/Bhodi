from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry
from bhodi_platform.indexing.infrastructure import build_retriever, build_vectorstore
from bhodi_platform.indexing.settings import IndexingSettings
from bhodi_platform.retrieval.runtime import get_embeddings


EmbeddingsFactory = Callable[[], Any]


def initialize_persistent_runtime(
    persist_directory: str,
    embeddings_factory: EmbeddingsFactory | None = None,
) -> tuple[Any, Any]:
    settings = IndexingSettings(persist_directory=Path(persist_directory))
    embeddings = (embeddings_factory or get_embeddings)()
    vectorstore = build_vectorstore(settings, embeddings)
    retriever = build_retriever(vectorstore, settings)
    return vectorstore, retriever


_persistent_runtime_registry = RuntimeRegistry(
    lambda: initialize_persistent_runtime(
        str(IndexingSettings.from_environment().persist_directory)
    )
)


def start_persistent_runtime() -> tuple[Any, Any]:
    return _persistent_runtime_registry.start()


def get_persistent_runtime() -> tuple[Any, Any]:
    return _persistent_runtime_registry.get()


def reset_persistent_runtime() -> None:
    _persistent_runtime_registry.reset()


def stop_persistent_runtime() -> None:
    _persistent_runtime_registry.stop()


def get_persistent_vectorstore() -> Any:
    return get_persistent_runtime()[0]


def get_persistent_retriever() -> Any:
    return get_persistent_runtime()[1]
