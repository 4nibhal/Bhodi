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
    "AnswerQueryUseCase",
    "BhodiApplication",
    "BhodiRuntime",
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
    "AnswerQueryUseCase": (
        "bhodi_platform.application.answer_query",
        "AnswerQueryUseCase",
    ),
    "BhodiApplication": (
        "bhodi_platform.application.facade",
        "BhodiApplication",
    ),
    "BhodiRuntime": (
        "bhodi_platform.application.runtime",
        "BhodiRuntime",
    ),
    "IndexDocumentsUseCase": (
        "bhodi_platform.application.index_documents",
        "IndexDocumentsUseCase",
    ),
}

if TYPE_CHECKING:
    from bhodi_platform.application.answer_query import AnswerQueryUseCase
    from bhodi_platform.application.facade import BhodiApplication
    from bhodi_platform.application.index_documents import IndexDocumentsUseCase
    from bhodi_platform.application.runtime import BhodiRuntime


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)
