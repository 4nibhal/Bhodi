"""Unit tests for the Bhodi application facade."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.application.models import IndexDocumentRequest, QueryRequest
from bhodi_platform.domain.entities import Chunk, Document, RetrievedDocument
from bhodi_platform.domain.value_objects import ChunkId, DocumentId


def _build_app(
    *,
    embedding: AsyncMock | None = None,
    vector_store: AsyncMock | None = None,
    chunker: AsyncMock | None = None,
    document_parser: AsyncMock | None = None,
    llm: AsyncMock | None = None,
    conversation_memory: AsyncMock | None = None,
) -> BhodiApplication:
    return BhodiApplication(
        embedding=embedding or AsyncMock(),
        vector_store=vector_store or AsyncMock(),
        chunker=chunker or AsyncMock(),
        document_parser=document_parser or AsyncMock(),
        llm=llm or AsyncMock(),
        conversation_memory=conversation_memory or AsyncMock(),
    )


@pytest.mark.asyncio
async def test_index_document_preserves_document_identity_and_merges_metadata():
    """Indexing should preserve parser identity and authoritative provenance."""
    parser_doc_id = DocumentId("11111111-1111-1111-1111-111111111111")
    chunker_doc_id = DocumentId("22222222-2222-2222-2222-222222222222")
    parser_document = Document(
        id=parser_doc_id,
        text="alpha beta gamma delta",
        metadata={
            "source": "doc.pdf",
            "filename": "doc.pdf",
            "title": "Parser Title",
            "custom": "parser",
        },
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    parser = AsyncMock()
    parser.parse.return_value = parser_document

    chunker = AsyncMock()
    chunker.chunk.return_value = [
        Chunk(
            id=ChunkId(document_id=chunker_doc_id, chunk_index=0),
            document_id=chunker_doc_id,
            content="alpha beta",
            chunk_index=0,
            total_chunks=2,
            metadata={"page": 3, "section": "intro"},
        ),
        Chunk(
            id=ChunkId(document_id=chunker_doc_id, chunk_index=1),
            document_id=chunker_doc_id,
            content="gamma delta",
            chunk_index=1,
            total_chunks=2,
            metadata={"page": 4},
        ),
    ]

    embedding = AsyncMock()
    embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    vector_store = AsyncMock()

    app = _build_app(
        embedding=embedding,
        vector_store=vector_store,
        chunker=chunker,
        document_parser=parser,
    )

    response = await app.index_document(
        IndexDocumentRequest(
            source="ignored.pdf",
            metadata={
                "source": "user-supplied-source",
                "title": "User Title",
                "custom": "user",
                "category": "policy",
            },
        )
    )

    assert response.document_id == str(parser_doc_id)
    assert response.chunk_count == 2

    stored_chunks, stored_embeddings = vector_store.add.await_args.args
    assert stored_embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert [chunk.document_id for chunk in stored_chunks] == [parser_doc_id, parser_doc_id]
    assert [str(chunk.id) for chunk in stored_chunks] == [
        f"{parser_doc_id}:0",
        f"{parser_doc_id}:1",
    ]
    assert all(chunk.total_chunks == 2 for chunk in stored_chunks)
    assert stored_chunks[0].metadata["source"] == "doc.pdf"
    assert stored_chunks[0].metadata["filename"] == "doc.pdf"
    assert stored_chunks[0].metadata["title"] == "Parser Title"
    assert stored_chunks[0].metadata["user_source"] == "user-supplied-source"
    assert stored_chunks[0].metadata["user_title"] == "User Title"
    assert stored_chunks[0].metadata["custom"] == "user"
    assert stored_chunks[0].metadata["category"] == "policy"
    assert stored_chunks[0].metadata["page"] == 3
    assert stored_chunks[0].metadata["section"] == "intro"


@pytest.mark.asyncio
async def test_query_uses_human_provenance_in_citations():
    """Query citations should prefer human-friendly provenance metadata."""
    document_id = DocumentId("33333333-3333-3333-3333-333333333333")
    retrieved = RetrievedDocument(
        chunk_id=ChunkId(document_id=document_id, chunk_index=0),
        document_id=document_id,
        text="Relevant excerpt from the document.",
        score=0.9,
        metadata={"filename": "doc.pdf", "page": "7"},
    )

    embedding = AsyncMock()
    embedding.embed_query.return_value = [0.1, 0.2]
    vector_store = AsyncMock()
    vector_store.search.return_value = [retrieved]
    llm = AsyncMock()
    llm.generate_with_context.return_value = "Answer"

    app = _build_app(embedding=embedding, vector_store=vector_store, llm=llm)

    response = await app.query(QueryRequest(question="What happened?", top_k=1))

    assert response.answer_text == "Answer"
    assert response.citations[0].source_document == "doc.pdf"
    assert response.citations[0].page == 7


@pytest.mark.asyncio
async def test_health_check_is_non_invasive():
    """Health checks should not invoke external adapters."""
    embedding = AsyncMock()
    embedding.embed_query.side_effect = AssertionError("should not be called")
    vector_store = AsyncMock()
    vector_store.persist.side_effect = AssertionError("should not be called")
    llm = AsyncMock()
    llm.generate.side_effect = AssertionError("should not be called")

    app = _build_app(embedding=embedding, vector_store=vector_store, llm=llm)

    health = await app.health_check()

    assert health.status == "healthy"
    assert health.version == "1.0.0"
    assert health.services == {"embedding": True, "vector_store": True, "llm": True}
    embedding.embed_query.assert_not_called()
    vector_store.persist.assert_not_called()
    llm.generate.assert_not_called()
