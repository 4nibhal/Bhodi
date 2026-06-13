"""Unit tests for the bodhi-rag application facade."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from bodhi_rag._version import get_version
from bodhi_rag.application.facade import BhodiApplication
from bodhi_rag.application.models import IndexDocumentRequest, QueryRequest
from bodhi_rag.domain.entities import Chunk, Document, RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId


def _build_app(
    *,
    embedding: AsyncMock | None = None,
    vector_store: AsyncMock | None = None,
    chunker: AsyncMock | None = None,
    document_parser: AsyncMock | None = None,
    llm: AsyncMock | None = None,
    conversation_memory: AsyncMock | None = None,
    reranker: AsyncMock | None = None,
) -> BhodiApplication:
    reranker_mock = reranker or AsyncMock()
    # Wave 3b: facade calls `self._reranker.overfetch_factor` (an int) to
    # compute the overfetched retrieval size, then `self._reranker.rerank(...)`
    # which must return an iterable. Configure the default mock so tests
    # that don't care about reranking still work end-to-end.
    if not reranker:
        reranker_mock.overfetch_factor = 1
        reranker_mock.rerank.side_effect = (
            lambda _query, chunks, top_k=None: list(chunks if top_k is None else chunks[:top_k])
        )
    return BhodiApplication(
        embedding=embedding or AsyncMock(),
        vector_store=vector_store or AsyncMock(),
        chunker=chunker or AsyncMock(),
        document_parser=document_parser or AsyncMock(),
        llm=llm or AsyncMock(),
        conversation_memory=conversation_memory or AsyncMock(),
        reranker=reranker_mock,
    )


@pytest.mark.asyncio
async def test_index_document_preserves_document_identity_and_merges_metadata() -> None:
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
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
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
        ),
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
async def test_query_uses_human_provenance_in_citations() -> None:
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
async def test_health_check_is_non_invasive() -> None:
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
    assert health.version == get_version()
    assert health.services == {"embedding": True, "vector_store": True, "llm": True}
    embedding.embed_query.assert_not_called()
    vector_store.persist.assert_not_called()


# --- Wave 3b: reranker integration -------------------------------------


@pytest.mark.asyncio
async def test_query_overfetches_then_reranks_then_trims_to_top_k() -> None:
    """Overfetch from the store, rerank the candidates, trim to top_k, then hand off."""
    doc_id = DocumentId("44444444-4444-4444-4444-444444444444")
    candidates = [
        RetrievedDocument(
            chunk_id=ChunkId(document_id=doc_id, chunk_index=i),
            document_id=doc_id,
            text=f"chunk {i}",
            score=float(i),
            metadata={"idx": i},
        )
        for i in range(8)
    ]
    # Reranker "promotes" idx=5 to the front and trims to top_k=2.
    reranked = [candidates[5], candidates[2]]

    embedding = AsyncMock()
    embedding.embed_query.return_value = [0.1]
    vector_store = AsyncMock()
    vector_store.search.return_value = candidates
    reranker = AsyncMock()
    reranker.overfetch_factor = 4
    reranker.rerank.return_value = reranked
    llm = AsyncMock()
    llm.generate_with_context.return_value = "promoted answer"

    app = _build_app(
        embedding=embedding,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
    )

    response = await app.query(QueryRequest(question="q", top_k=2))

    # The vector store must be asked for top_k * overfetch_factor.
    vector_store.search.assert_awaited_once()
    assert vector_store.search.await_args.args[1] == 8  # 2 * 4

    # The reranker must receive the overfetched candidates and trim to top_k.
    reranker.rerank.assert_awaited_once()
    args, kwargs = reranker.rerank.await_args
    assert args[0] == "q"
    assert args[1] == candidates
    assert kwargs["top_k"] == 2

    # The LLM and the citations must use the reranked list, not the raw one.
    assert llm.generate_with_context.await_args.args[1] == reranked
    assert [c.text for c in response.citations] == ["chunk 5", "chunk 2"]


@pytest.mark.asyncio
async def test_query_uses_noop_passthrough_when_overfetch_factor_is_one() -> None:
    """NoOpReranker with overfetch_factor=1 preserves pre-Wave-3b behavior exactly."""
    doc_id = DocumentId("55555555-5555-5555-5555-555555555555")
    chunks = [
        RetrievedDocument(
            chunk_id=ChunkId(document_id=doc_id, chunk_index=i),
            document_id=doc_id,
            text=f"chunk {i}",
            score=float(i),
            metadata={"idx": i},
        )
        for i in range(3)
    ]

    embedding = AsyncMock()
    embedding.embed_query.return_value = [0.1]
    vector_store = AsyncMock()
    vector_store.search.return_value = chunks
    reranker = AsyncMock()
    reranker.overfetch_factor = 1
    reranker.rerank.side_effect = (
        lambda _q, c, top_k=None: list(c if top_k is None else c[:top_k])
    )
    llm = AsyncMock()
    llm.generate_with_context.return_value = "answer"

    app = _build_app(
        embedding=embedding,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
    )

    response = await app.query(QueryRequest(question="q", top_k=3))

    # Vector store asked for exactly top_k (no overfetch).
    assert vector_store.search.await_args.args[1] == 3
    # Reranker trimmed to top_k (no-op truncation).
    assert len(response.citations) == 3
    assert [c.text for c in response.citations] == ["chunk 0", "chunk 1", "chunk 2"]


@pytest.mark.asyncio
async def test_query_does_not_overfetch_below_top_k() -> None:
    """overfetch_factor=0 must still yield at least top_k from the store (no under-fetch)."""
    embedding = AsyncMock()
    embedding.embed_query.return_value = [0.1]
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    reranker = AsyncMock()
    reranker.overfetch_factor = 0
    reranker.rerank.return_value = []
    llm = AsyncMock()
    llm.generate_with_context.return_value = ""

    app = _build_app(
        embedding=embedding,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
    )

    await app.query(QueryRequest(question="q", top_k=5))

    assert vector_store.search.await_args.args[1] == 5

    llm.generate.assert_not_called()
