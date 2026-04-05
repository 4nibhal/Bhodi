"""
In-memory vector store adapter for testing.

Provides a simple in-memory implementation for tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import VectorStoreConfig
    from bhodi_platform.domain.entities import Chunk
    from bhodi_platform.domain.value_objects import DocumentId


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
        for chunk, embedding in zip(chunks, embeddings):
            key = str(chunk.id)
            self._chunks[key] = (chunk, embedding)

    @traced("mock.vector_store.search")
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list:
        """Simple search - return first top_k chunks with fake scores."""
        from bhodi_platform.domain.entities import RetrievedDocument

        results = []
        for key, (chunk, embedding) in self._chunks.items():
            # Fake score based on first dimension
            score = embedding[0] if embedding else 0.0
            results.append(
                RetrievedDocument(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                )
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
        for key in keys_to_delete:
            del self._chunks[key]

    async def persist(self) -> None:
        """No-op for in-memory store."""
        pass
