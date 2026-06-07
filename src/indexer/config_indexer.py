"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from bhodi_platform.indexing import (
    EmbeddingsFactory,
    IndexingSettings,
    get_persistent_retriever,
    get_persistent_vectorstore,
    initialize_persistent_runtime,
)

FactoryCallable: TypeAlias = Callable[[], Any]


def initialize_vectorstore(
    persist_directory: str,
    embeddings_factory: EmbeddingsFactory | None = None,
) -> tuple[Any, Any]:
    """
    Initialize the persistent embeddings, vectorstore, and retriever.

    Args:
        persist_directory: Path where the vectorstore will persist data.
        embeddings_factory: Optional embeddings factory override.

    Returns:
        tuple[Any, Any]: The delegated vectorstore and retriever.

    """
    return initialize_persistent_runtime(
        persist_directory,
        embeddings_factory=embeddings_factory,
    )


class _LazyObjectProxy:
    def __init__(self, factory: FactoryCallable) -> None:
        self._factory = factory

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._factory(), attribute)


PERSIST_DIRECTORY = str(IndexingSettings.from_environment().persist_directory)
persistent_vectorstore = _LazyObjectProxy(get_persistent_vectorstore)
persistent_retriever = _LazyObjectProxy(get_persistent_retriever)

__all__ = [
    "PERSIST_DIRECTORY",
    "initialize_vectorstore",
    "persistent_retriever",
    "persistent_vectorstore",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return locals()[name]
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
