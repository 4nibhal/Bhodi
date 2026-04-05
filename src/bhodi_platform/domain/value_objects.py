"""Domain value objects for Bhodi platform.

Immutable, frozen dataclasses that represent descriptive aspects
of the domain with no conceptual identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class DocumentId:
    """Unique identifier for a document.

    Uses UUID4 for uniqueness.
    """

    value: UUID

    def __init__(self, value: str | UUID | None = None) -> None:
        if value is None:
            object.__setattr__(self, "value", uuid4())
        elif isinstance(value, UUID):
            object.__setattr__(self, "value", value)
        else:
            object.__setattr__(self, "value", UUID(value))

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"DocumentId({self.value})"


@dataclass(frozen=True, slots=True)
class ChunkId:
    """Unique identifier for a document chunk.

    Composed of document ID + chunk index for uniqueness.
    """

    document_id: DocumentId
    chunk_index: int

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")

    def __str__(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"

    def __repr__(self) -> str:
        return f"ChunkId({self.document_id}:{self.chunk_index})"


@dataclass(frozen=True, slots=True)
class ConversationId:
    """Unique identifier for a conversation session."""

    value: UUID

    def __init__(self, value: str | UUID | None = None) -> None:
        if value is None:
            object.__setattr__(self, "value", uuid4())
        elif isinstance(value, UUID):
            object.__setattr__(self, "value", value)
        else:
            object.__setattr__(self, "value", UUID(value))

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"ConversationId({self.value})"


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """An embedding vector as an immutable sequence of floats."""

    values: Sequence[float]

    def __init__(self, values: Sequence[float]) -> None:
        object.__setattr__(self, "values", tuple(values))

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingVector):
            return NotImplemented
        return self.values == other.values

    def __hash__(self) -> int:
        return hash(self.values)


@dataclass(frozen=True, slots=True)
class Citation:
    """A reference to a source chunk with position information.

    Used for answer citations.
    """

    chunk_id: ChunkId
    text: str
    source_document: str
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": str(self.chunk_id),
            "text": self.text,
            "source_document": self.source_document,
            "page": self.page,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True, slots=True)
class DocumentOrigin:
    """Tracks where a document came from in the retrieval pipeline."""

    retriever_type: str  # "session", "conversation", "corpus"
    source: str | None = None  # file path or memory URI
    chunk_index: int | None = None

    def __post_init__(self) -> None:
        if self.retriever_type not in ("session", "conversation", "corpus"):
            raise ValueError(
                f"Invalid retriever_type: {self.retriever_type}. "
                "Must be 'session', 'conversation', or 'corpus'."
            )


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata associated with a document chunk."""

    source: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    document_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.source is not None:
            result["source"] = self.source
        if self.chunk_index is not None:
            result["chunk_index"] = self.chunk_index
        if self.total_chunks is not None:
            result["total_chunks"] = self.total_chunks
        if self.document_type is not None:
            result["document_type"] = self.document_type
        result.update(self.extra)
        return result


@dataclass(frozen=True, slots=True)
class AnswerMetadata:
    """Metadata associated with a generated answer."""

    model: str | None = None
    generation_latency_ms: int | None = None
    tokens_used: int | None = None
    prompt_version: str | None = None


@dataclass(frozen=True, slots=True)
class TruncationDiagnostics:
    """Diagnostics for truncation decisions in retrieval."""

    original_length: int
    returned_length: int
    truncated: bool
    truncation_type: str | None = None  # "token_budget", "character", "summarization"

    def __post_init__(self) -> None:
        if self.truncation_type not in (
            None,
            "token_budget",
            "character",
            "summarization",
        ):
            raise ValueError(f"Invalid truncation_type: {self.truncation_type}")
