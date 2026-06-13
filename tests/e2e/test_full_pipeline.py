"""Minimal E2E tests for indexing and query pipeline."""

import pytest

from bodhi_rag._version import get_version
from bodhi_rag.application.config import (
    BhodiConfig,
    ChunkerConfig,
    ConversationConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from bodhi_rag.application.facade import BhodiApplication
from bodhi_rag.application.models import IndexDocumentRequest, QueryRequest
from bodhi_rag.infrastructure.container import Container


@pytest.fixture
def app() -> BhodiApplication:
    """Build the app through the real Container (F5-B)."""
    config = BhodiConfig(
        embedding=EmbeddingConfig(provider="mock", dimensions=16),
        vector_store=VectorStoreConfig(provider="in_memory"),
        chunker=ChunkerConfig(provider="fixed_size"),
        llm=LLMConfig(provider="mock"),
        conversation=ConversationConfig(provider="volatile"),
        parser={"provider": "mock"},
    )
    return Container(config).build()


@pytest.mark.asyncio
async def test_health_check(app: BhodiApplication) -> None:
    """Health check should return healthy status."""
    status = await app.health_check()
    assert status.status == "healthy"
    assert status.version == get_version()


@pytest.mark.asyncio
async def test_index_document(app: BhodiApplication) -> None:
    """Index a document and get back document ID."""
    request = IndexDocumentRequest(source="test.pdf")
    response = await app.index_document(request)

    assert response.document_id is not None
    assert response.chunk_count >= 1


@pytest.mark.asyncio
async def test_query(app: BhodiApplication) -> None:
    """Query the index after indexing a document."""
    # Index first
    await app.index_document(IndexDocumentRequest(source="test.pdf"))

    # Then query
    request = QueryRequest(question="What is this about?", top_k=3)
    response = await app.query(request)

    assert response.answer_text is not None
    assert len(response.answer_text) > 0
