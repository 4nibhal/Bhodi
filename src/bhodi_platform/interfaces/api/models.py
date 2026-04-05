from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bhodi_platform.application import (
    AnswerQueryResponse,
    RetrievedDocumentDiagnostics,
    RetrievalDiagnostics,
    RetrievalTruncationDiagnostics,
)


class ApiConversationMessage(BaseModel):
    role: str
    content: str


class QueryRequestModel(BaseModel):
    user_input: str = Field(..., min_length=1)
    messages: list[ApiConversationMessage] = Field(default_factory=list)
    conversation_id: str | None = None


class DocumentIndexRequestModel(BaseModel):
    document_path: str = Field(..., min_length=1)
    cwd: str | None = None


class RetrievalTruncationModel(BaseModel):
    original_length: int
    returned_length: int
    truncated: bool

    @classmethod
    def from_domain(
        cls, diagnostics: RetrievalTruncationDiagnostics
    ) -> "RetrievalTruncationModel":
        return cls(
            original_length=diagnostics.original_length,
            returned_length=diagnostics.returned_length,
            truncated=diagnostics.truncated,
        )


class RetrievedDocumentModel(BaseModel):
    rank: int
    retriever_origin: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    summarized: bool
    used_in_context: bool
    preview: str
    truncation: RetrievalTruncationModel

    @classmethod
    def from_domain(
        cls, diagnostics: RetrievedDocumentDiagnostics
    ) -> "RetrievedDocumentModel":
        return cls(
            rank=diagnostics.rank,
            retriever_origin=diagnostics.retriever_origin,
            source=diagnostics.source,
            metadata=diagnostics.metadata,
            summarized=diagnostics.summarized,
            used_in_context=diagnostics.used_in_context,
            preview=diagnostics.preview,
            truncation=RetrievalTruncationModel.from_domain(diagnostics.truncation),
        )


class RetrievalDiagnosticsModel(BaseModel):
    documents: list[RetrievedDocumentModel] = Field(default_factory=list)
    context: RetrievalTruncationModel

    @classmethod
    def from_domain(
        cls, diagnostics: RetrievalDiagnostics
    ) -> "RetrievalDiagnosticsModel":
        return cls(
            documents=[
                RetrievedDocumentModel.from_domain(document)
                for document in diagnostics.documents
            ],
            context=RetrievalTruncationModel.from_domain(diagnostics.context),
        )


class QueryResponseModel(BaseModel):
    answer_text: str
    context: str
    retrieval: RetrievalDiagnosticsModel | None = None

    @classmethod
    def from_domain(cls, response: AnswerQueryResponse) -> "QueryResponseModel":
        return cls(
            answer_text=response.answer_text,
            context=response.context,
            retrieval=(
                RetrievalDiagnosticsModel.from_domain(response.retrieval)
                if response.retrieval is not None
                else None
            ),
        )


class DocumentIndexResponseModel(BaseModel):
    indexed_fragments: int
    source_kind: str
    resolved_path: str


class HealthResponseModel(BaseModel):
    status: str
    service: str
    runtime: dict[str, bool]


def resolve_request_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value)
