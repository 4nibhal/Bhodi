from importlib import import_module
from typing import TYPE_CHECKING, Any

from bhodi_platform.application.models import (
    CitationResponse,
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    QueryRequest,
    QueryResponse,
)

__all__ = [
    "BhodiApplication",
    "CitationResponse",
    "HealthStatus",
    "IndexDocumentRequest",
    "IndexDocumentResponse",
    "IndexDocumentsRequest",
    "IndexDocumentsResponse",
    "IndexDocumentsUseCase",
    "QueryRequest",
    "QueryResponse",
]

_EXPORTS = {
    "BhodiApplication": (
        "bhodi_platform.application.facade",
        "BhodiApplication",
    ),
    "IndexDocumentsUseCase": (
        "bhodi_platform.application.index_documents",
        "IndexDocumentsUseCase",
    ),
}

if TYPE_CHECKING:
    from bhodi_platform.application.facade import BhodiApplication
    from bhodi_platform.application.index_documents import IndexDocumentsUseCase


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)
