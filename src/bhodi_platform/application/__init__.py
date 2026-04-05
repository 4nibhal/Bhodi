from bhodi_platform.application.models import (
    CitationResponse,
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
)

__all__ = [
    "AnswerQueryUseCase",
    "BhodiApplication",
    "CitationResponse",
    "HealthStatus",
    "IndexDocumentRequest",
    "IndexDocumentResponse",
    "IndexDocumentsUseCase",
    "QueryRequest",
    "QueryResponse",
    "BhodiRuntime",
]


def __getattr__(name: str):
    if name == "AnswerQueryUseCase":
        from bhodi_platform.application.answer_query import AnswerQueryUseCase

        return AnswerQueryUseCase
    if name == "BhodiApplication":
        from bhodi_platform.application.facade import BhodiApplication

        return BhodiApplication
    if name == "IndexDocumentsUseCase":
        from bhodi_platform.application.index_documents import IndexDocumentsUseCase

        return IndexDocumentsUseCase
    if name == "BhodiRuntime":
        from bhodi_platform.application.runtime import BhodiRuntime

        return BhodiRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
