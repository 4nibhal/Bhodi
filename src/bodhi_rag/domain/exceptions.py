"""Domain exceptions for bodhi-rag platform."""

from __future__ import annotations


class BodhiRagDomainError(Exception):
    """Base exception for all domain errors."""

    pass


class DomainValidationError(BodhiRagDomainError):
    """Raised when domain invariants or validation rules are violated."""

    pass


class PolicyViolationError(BodhiRagDomainError):
    """Raised when an action violates defined business policies."""

    pass


class DocumentIntegrityError(BodhiRagDomainError):
    """Raised when a document fails integrity checks (e.g., missing required fields)."""

    pass


class DocumentNotFoundError(BodhiRagDomainError):
    """Raised when a requested document does not exist in the store."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class ChunkingError(BodhiRagDomainError):
    """Raised when text chunking fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Chunking failed: {reason}")


class EmbeddingError(BodhiRagDomainError):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Embedding generation failed: {reason}")


class VectorStoreError(BodhiRagDomainError):
    """Raised for vector storage and retrieval failures."""

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store {operation} failed: {reason}")


class InvalidDocumentError(BodhiRagDomainError):
    """Raised when document format is invalid or cannot be parsed."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid document: {reason}")


class LLMError(BodhiRagDomainError):
    """Raised when LLM generation fails."""

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"LLM {operation} failed: {reason}")
