"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bhodi_platform.indexing.runtime import (
    EmbeddingsFactory,
    get_persistent_retriever,
    get_persistent_runtime,
    get_persistent_vectorstore,
    initialize_persistent_runtime,
)
from bhodi_platform.indexing.settings import IndexingSettings


def initialize_vectorstore(
    persist_directory: str,
    embeddings_factory: EmbeddingsFactory | None = None,
) -> tuple[Any, Any]:
    """
    Initializes the embeddings, vectorstore, and retriever for document indexing.
    Uses persistent storage.

    Args:
        persist_directory (str): Path where the vectorstore will persist data.

    Returns:
        Tuple containing the vectorstore and retriever.
    """
    return initialize_persistent_runtime(
        persist_directory,
        embeddings_factory=embeddings_factory,
    )


class _LazyObjectProxy:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._factory(), attribute)


PERSIST_DIRECTORY = str(IndexingSettings.from_environment().persist_directory)
persistent_vectorstore = _LazyObjectProxy(get_persistent_vectorstore)
persistent_retriever = _LazyObjectProxy(get_persistent_retriever)

__all__ = [
    "PERSIST_DIRECTORY",
    "initialize_vectorstore",
    "persistent_vectorstore",
    "persistent_retriever",
]


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
