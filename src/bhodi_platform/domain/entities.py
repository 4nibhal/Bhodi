"""Domain entities for Bhodi platform.

These are simple, Pythonic frozen dataclasses with slots.
No heavy ORM patterns - pure data with business meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bhodi_platform.domain.value_objects import (
    Citation,
    ChunkId,
    ConversationId,
    DocumentId,
)


@dataclass(frozen=True, slots=True)
class Document:
    """Represents a document to be indexed.

    Contains the raw content and metadata.
    """

    id: DocumentId
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    indexed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Document text cannot be empty")


@dataclass(frozen=True, slots=True)
class Query:
    """Represents a user query in the retrieval pipeline."""

    text: str
    conversation_id: ConversationId | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Query text cannot be empty")


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """Represents a document retrieved from the vector store.

    Contains the chunk text and scoring information.
    """

    chunk_id: ChunkId
    document_id: DocumentId
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """Represents a chunk of a document for indexing."""

    id: ChunkId
    document_id: DocumentId
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.chunk_index >= self.total_chunks:
            raise ValueError("chunk_index must be less than total_chunks")


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Represents a document that has been indexed."""

    source: str
    chunk_count: int
    indexed_at: str  # ISO timestamp
    chunks: tuple[Chunk, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Answer:
    """Represents a generated answer with citations."""

    text: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Represents a single turn in a conversation."""

    conversation_id: ConversationId
    user_message: str
    assistant_message: str
    turn_index: int = 0
    citations: tuple[Citation, ...] = field(default_factory=tuple)
