"""Domain layer for Bhodi platform.

This package contains pure business logic with no infrastructure dependencies.
All entities, value objects, policies, and domain services live here.

Exports:
    Entities: Query, RetrievedDocument, Answer, ConversationTurn, Chunk, IndexedDocument
    Value Objects: DocumentOrigin, ChunkMetadata, AnswerMetadata, TruncationDiagnostics
    Policies: RetrievalPolicy, GenerationPolicy, ContextAssemblyPolicy, IndexingPolicy
    Exceptions: DomainValidationError, PolicyViolationError, DocumentIntegrityError
    Services: RetrievalDomainService, IndexingDomainService
"""

from bhodi_platform.domain.entities import (
    Answer,
    Chunk,
    ConversationTurn,
    IndexedDocument,
    Query,
    RetrievedDocument,
)
from bhodi_platform.domain.exceptions import (
    BhodiDomainError,
    DomainValidationError,
    DocumentIntegrityError,
    PolicyViolationError,
)
from bhodi_platform.domain.policies import (
    ContextAssemblyPolicy,
    GenerationPolicy,
    IndexingPolicy,
    RetrievalPolicy,
)
from bhodi_platform.domain.services import (
    IndexingDomainService,
    RetrievalDomainService,
)
from bhodi_platform.domain.value_objects import (
    AnswerMetadata,
    ChunkMetadata,
    DocumentOrigin,
    TruncationDiagnostics,
)

__all__ = [
    # Entities
    "Answer",
    "Chunk",
    "ConversationTurn",
    "IndexedDocument",
    "Query",
    "RetrievedDocument",
    # Value Objects
    "AnswerMetadata",
    "ChunkMetadata",
    "DocumentOrigin",
    "TruncationDiagnostics",
    # Policies
    "ContextAssemblyPolicy",
    "GenerationPolicy",
    "IndexingPolicy",
    "RetrievalPolicy",
    # Exceptions
    "BhodiDomainError",
    "DomainValidationError",
    "DocumentIntegrityError",
    "PolicyViolationError",
    # Services
    "IndexingDomainService",
    "RetrievalDomainService",
]
