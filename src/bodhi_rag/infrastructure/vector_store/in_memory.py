"""
In-memory vector store adapter for testing.

Provides a simple in-memory implementation for tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bodhi_rag.domain.exceptions import DocumentNotFoundError
from bodhi_rag.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bodhi_rag.application.config import VectorStoreConfig
    from bodhi_rag.domain.entities import Chunk
    from bodhi_rag.domain.value_objects import DocumentId


class MockVectorStoreAdapter:
    """
    In-memory vector store for testing.

    Not suitable for production use.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._chunks: dict[str, tuple[Chunk, list[float]]] = {}

    @traced("mock.vector_store.add")
    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks with their embeddings."""
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            key = str(chunk.id)
            self._chunks[key] = (chunk, embedding)

    @traced("mock.vector_store.search")
    async def search(
        self,
        _query_embedding: list[float],
        top_k: int,
    ) -> list:
        """Return first top_k chunks with fake scores."""
        from bodhi_rag.domain.entities import RetrievedDocument

        results = []
        for _key, (chunk, embedding) in self._chunks.items():
            # Fake score based on first dimension
            score = embedding[0] if embedding else 0.0
            results.append(
                RetrievedDocument(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                ),
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score or 0.0, reverse=True)
        return results[:top_k]

    @traced("mock.vector_store.delete")
    async def delete(self, document_id: DocumentId) -> None:
        """Delete all chunks for a document."""
        keys_to_delete = [
            k
            for k, (chunk, _) in self._chunks.items()
            if str(chunk.document_id) == str(document_id)
        ]
        if not keys_to_delete:
            raise DocumentNotFoundError(str(document_id))
        for key in keys_to_delete:
            del self._chunks[key]

    async def persist(self) -> None:
        """No-op for in-memory store."""
