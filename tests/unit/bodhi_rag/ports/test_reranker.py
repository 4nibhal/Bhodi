"""Contract test for the RerankerPort Protocol."""

from __future__ import annotations

import asyncio

from bodhi_rag.domain.entities import RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.ports.reranker import RerankerPort


def _make_chunk(idx: int) -> RetrievedDocument:
    doc_id = DocumentId()
    return RetrievedDocument(
        chunk_id=ChunkId(document_id=doc_id, chunk_index=idx),
        document_id=doc_id,
        text=f"chunk text {idx}",
        score=1.0 - (idx * 0.1),
        metadata={"idx": idx},
    )


class _StubReranker:
    """Minimal stand-in that implements the RerankerPort shape."""

    async def rerank(
        self,
        query: str,  # noqa: ARG002
        chunks: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        if top_k is None:
            return list(chunks)
        return list(chunks[:top_k])


def test_reranker_port_is_runtime_checkable() -> None:
    """A class that exposes `async def rerank(...)` should satisfy RerankerPort."""
    assert isinstance(_StubReranker(), RerankerPort)


def test_reranker_port_protocol_annotates_inputs() -> None:
    """The protocol declares the expected signature; consumers rely on it."""
    hints = RerankerPort.rerank.__annotations__
    # `from __future__ import annotations` makes annotations strings at runtime.
    assert hints["query"] == "str"
    assert "chunks" in hints
    assert "top_k" in hints
    assert "return" in hints


def test_rerank_input_chunks_are_preserved_in_count() -> None:
    """The contract does not require deduplication or truncation by default."""
    chunks = [_make_chunk(i) for i in range(5)]
    stub = _StubReranker()
    result = asyncio.run(stub.rerank("query", chunks))
    assert len(result) == 5
