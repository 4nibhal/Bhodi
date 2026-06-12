"""Backend-first indexing slice for bodhi-rag."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "IndexDocumentsRequest",
    "IndexDocumentsResponse",
    "IndexingSettings",
    "load_documents_from_directory",
    "load_documents_from_file",
]

_EXPORTS = {
    "IndexDocumentsRequest": (
        "bodhi_rag.application.models",
        "IndexDocumentsRequest",
    ),
    "IndexDocumentsResponse": (
        "bodhi_rag.application.models",
        "IndexDocumentsResponse",
    ),
    "IndexingSettings": ("bodhi_rag.indexing.settings", "IndexingSettings"),
    "load_documents_from_directory": (
        "bodhi_rag.indexing.infrastructure",
        "load_documents_from_directory",
    ),
    "load_documents_from_file": (
        "bodhi_rag.indexing.infrastructure",
        "load_documents_from_file",
    ),
}

if TYPE_CHECKING:
    from bodhi_rag.application.models import (
        IndexDocumentsRequest,
        IndexDocumentsResponse,
    )
    from bodhi_rag.indexing.settings import IndexingSettings


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
