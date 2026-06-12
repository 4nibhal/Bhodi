"""Domain layer for bodhi-rag platform.

This package contains pure business logic with no infrastructure dependencies.
All entities, value objects, policies, and domain services live here.

Exports:
    Entities: Query, RetrievedDocument, Answer, ConversationTurn, Chunk, IndexedDocument
    Value Objects: DocumentOrigin, ChunkMetadata, AnswerMetadata, TruncationDiagnostics
    Policies: RetrievalPolicy, GenerationPolicy, ContextAssemblyPolicy, IndexingPolicy
    Exceptions: DomainValidationError, PolicyViolationError, DocumentIntegrityError
    Services: RetrievalDomainService, IndexingDomainService
"""

from bodhi_rag.domain.entities import (
    Answer,
    Chunk,
    ConversationTurn,
    IndexedDocument,
    Query,
    RetrievedDocument,
)
from bodhi_rag.domain.exceptions import (
    BodhiRagDomainError,
    DomainValidationError,
    DocumentIntegrityError,
    PolicyViolationError,
)
from bodhi_rag.domain.policies import (
    ContextAssemblyPolicy,
    GenerationPolicy,
    IndexingPolicy,
    RetrievalPolicy,
)
from bodhi_rag.domain.services import (
    IndexingDomainService,
    RetrievalDomainService,
)
from bodhi_rag.domain.value_objects import (
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
    "BodhiRagDomainError",
    "DomainValidationError",
    "DocumentIntegrityError",
    "PolicyViolationError",
    # Services
    "IndexingDomainService",
    "RetrievalDomainService",
]
