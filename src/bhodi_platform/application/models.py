"""Request/Response models for Bhodi application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class IndexDocumentRequest(BaseModel):
    source: str | Path
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int | None = None
    overlap: int | None = None


class IndexDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class CitationResponse(BaseModel):
    chunk_id: str
    text: str
    source_document: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer_text: str
    citations: list[CitationResponse]
    conversation_id: str | None


class HealthStatus(BaseModel):
    status: str
    version: str
