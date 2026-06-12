from importlib import import_module
from typing import TYPE_CHECKING, Any

from bodhi_rag.application.models import (
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
    "QueryRequest",
    "QueryResponse",
]

_EXPORTS = {
    "BhodiApplication": (
        "bodhi_rag.application.facade",
        "BhodiApplication",
    ),
}

if TYPE_CHECKING:
    from bodhi_rag.application.facade import BhodiApplication


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)
