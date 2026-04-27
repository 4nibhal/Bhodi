"""Minimal E2E tests for indexing and query pipeline."""

import pytest

from bhodi_platform.application.config import (
    BhodiConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkerConfig,
    LLMConfig,
    ConversationConfig,
)
from bhodi_platform.application.models import IndexDocumentRequest
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.infrastructure.embedding.mock import MockEmbeddingAdapter
from bhodi_platform.infrastructure.vector_store.in_memory import MockVectorStoreAdapter
from bhodi_platform.infrastructure.llm.mock import MockLLMAdapter
from bhodi_platform.infrastructure.conversation_memory.volatile import (
    VolatileConversationMemoryAdapter,
)
from bhodi_platform.domain.entities import Chunk, Document
from bhodi_platform.domain.value_objects import ChunkId, DocumentId


class SimpleChunkerAdapter:
    """Minimal chunker - returns single chunk."""

    def __init__(self, config: ChunkerConfig):
        pass

    @property
    def default_chunk_size(self) -> int:
        return 100

    @property
    def default_overlap(self) -> int:
        return 10

    async def chunk(self, text: str, chunk_size=None, overlap=None):
        doc_id = DocumentId()
        return [
            Chunk(
                id=ChunkId(document_id=doc_id, chunk_index=0),
                document_id=doc_id,
                content=text[:100],
                chunk_index=0,
                total_chunks=1,
            )
        ]


class SimpleParserAdapter:
    """Minimal parser - returns simple document."""

    async def parse(self, source):
        return Document(id=DocumentId(), text="Test document content")

    async def extract_text(self, source):
        return "Test document content"

    async def extract_metadata(self, source):
        return {}


@pytest.fixture
def app():
    config = BhodiConfig(
        embedding=EmbeddingConfig(provider="mock", dimensions=16),
        vector_store=VectorStoreConfig(provider="in_memory"),
        chunker=ChunkerConfig(provider="fixed_size"),
        llm=LLMConfig(provider="mock"),
        conversation=ConversationConfig(provider="volatile"),
    )
    return BhodiApplication(
        embedding=MockEmbeddingAdapter(config.embedding),
        vector_store=MockVectorStoreAdapter(config.vector_store),
        chunker=SimpleChunkerAdapter(config.chunker),
        document_parser=SimpleParserAdapter(),
        llm=MockLLMAdapter(config.llm),
        conversation_memory=VolatileConversationMemoryAdapter(config.conversation),
    )


@pytest.mark.asyncio
async def test_health_check(app):
    """Health check should return healthy status."""
    status = await app.health_check()
    assert status.status == "healthy"
    assert status.version == "1.0.0"


@pytest.mark.asyncio
async def test_index_document(app):
    """Index a document and get back document ID."""
    from bhodi_platform.application.models import IndexDocumentRequest

    request = IndexDocumentRequest(source="test.pdf")
    response = await app.index_document(request)

    assert response.document_id is not None
    assert response.chunk_count >= 1


@pytest.mark.asyncio
async def test_query(app):
    """Query the index after indexing a document."""
    from bhodi_platform.application.models import IndexDocumentRequest, QueryRequest

    # Index first
    await app.index_document(IndexDocumentRequest(source="test.pdf"))

    # Then query
    request = QueryRequest(question="What is this about?", top_k=3)
    response = await app.query(request)

    assert response.answer_text is not None
    assert len(response.answer_text) > 0
