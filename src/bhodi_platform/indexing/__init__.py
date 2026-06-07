"""Backend-first indexing slice for Bhodi."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DocumentIndexingService",
    "EmbeddingsFactory",
    "IndexDocumentsRequest",
    "IndexDocumentsResponse",
    "IndexingEngine",
    "IndexingSettings",
    "InvalidDocumentPathError",
    "get_persistent_retriever",
    "get_persistent_runtime",
    "get_persistent_vectorstore",
    "initialize_persistent_runtime",
    "load_documents_from_directory",
    "load_documents_from_file",
    "reset_persistent_runtime",
    "start_persistent_runtime",
    "stop_persistent_runtime",
]

_EXPORTS = {
    "DocumentIndexingService": (
        "bhodi_platform.indexing.application",
        "DocumentIndexingService",
    ),
    "EmbeddingsFactory": ("bhodi_platform.indexing.runtime", "EmbeddingsFactory"),
    "IndexDocumentsRequest": (
        "bhodi_platform.application.models",
        "IndexDocumentsRequest",
    ),
    "IndexDocumentsResponse": (
        "bhodi_platform.application.models",
        "IndexDocumentsResponse",
    ),
    "IndexingEngine": ("bhodi_platform.indexing.engine", "IndexingEngine"),
    "IndexingSettings": ("bhodi_platform.indexing.settings", "IndexingSettings"),
    "InvalidDocumentPathError": (
        "bhodi_platform.indexing.errors",
        "InvalidDocumentPathError",
    ),
    "get_persistent_retriever": (
        "bhodi_platform.indexing.runtime",
        "get_persistent_retriever",
    ),
    "get_persistent_runtime": (
        "bhodi_platform.indexing.runtime",
        "get_persistent_runtime",
    ),
    "get_persistent_vectorstore": (
        "bhodi_platform.indexing.runtime",
        "get_persistent_vectorstore",
    ),
    "initialize_persistent_runtime": (
        "bhodi_platform.indexing.runtime",
        "initialize_persistent_runtime",
    ),
    "load_documents_from_directory": (
        "bhodi_platform.indexing.infrastructure",
        "load_documents_from_directory",
    ),
    "load_documents_from_file": (
        "bhodi_platform.indexing.infrastructure",
        "load_documents_from_file",
    ),
    "reset_persistent_runtime": (
        "bhodi_platform.indexing.runtime",
        "reset_persistent_runtime",
    ),
    "start_persistent_runtime": (
        "bhodi_platform.indexing.runtime",
        "start_persistent_runtime",
    ),
    "stop_persistent_runtime": (
        "bhodi_platform.indexing.runtime",
        "stop_persistent_runtime",
    ),
}

if TYPE_CHECKING:
    from bhodi_platform.application.models import (
        IndexDocumentsRequest,
        IndexDocumentsResponse,
    )
    from bhodi_platform.indexing.application import DocumentIndexingService
    from bhodi_platform.indexing.engine import IndexingEngine
    from bhodi_platform.indexing.errors import InvalidDocumentPathError
    from bhodi_platform.indexing.runtime import EmbeddingsFactory
    from bhodi_platform.indexing.settings import IndexingSettings


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
