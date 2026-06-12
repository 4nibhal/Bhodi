"""Unit tests for the NoOpReranker adapter."""

from __future__ import annotations

import pytest

from bodhi_rag.domain.entities import RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.infrastructure.reranker.noop import NoOpReranker


def _make_chunk(idx: int) -> RetrievedDocument:
    doc_id = DocumentId()
    return RetrievedDocument(
        chunk_id=ChunkId(document_id=doc_id, chunk_index=idx),
        document_id=doc_id,
        text=f"chunk {idx}",
        score=float(idx),
        metadata={"idx": idx},
    )


@pytest.mark.asyncio
async def test_noop_returns_input_unchanged_when_top_k_is_none() -> None:
    """NoOpReranker must return the input chunks in their original order."""
    chunks = [_make_chunk(i) for i in range(3)]
    result = await NoOpReranker().rerank("query", chunks)
    assert result == chunks


@pytest.mark.asyncio
async def test_noop_truncates_to_top_k() -> None:
    """NoOpReranker must respect `top_k` and return only the first N chunks."""
    chunks = [_make_chunk(i) for i in range(5)]
    result = await NoOpReranker().rerank("query", chunks, top_k=2)
    assert len(result) == 2
    assert result == chunks[:2]


@pytest.mark.asyncio
async def test_noop_returns_empty_for_empty_input() -> None:
    """NoOpReranker must handle empty input without raising."""
    result = await NoOpReranker().rerank("query", [])
    assert result == []


@pytest.mark.asyncio
async def test_noop_ignores_query() -> None:
    """The query parameter must be present for protocol compatibility but unused."""
    chunks = [_make_chunk(0)]
    result = await NoOpReranker().rerank("any query at all", chunks)
    assert result == chunks
