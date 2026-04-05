"""
Chroma vector store adapter.

Persists vectors using ChromaDB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bhodi_platform.domain.entities import Chunk
from bhodi_platform.domain.value_objects import DocumentId
from bhodi_platform.ports.vector_store import VectorStorePort

if TYPE_CHECKING:
    from bhodi_platform.application.config import VectorStoreConfig


class ChromaVectorStoreAdapter:
    """
    ChromaDB adapter for vector storage and retrieval.

    Uses chromadb PersistentClient for persistence.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._client: Any | None = None
        self._collection: Any | None = None

    async def _ensure_client(self) -> None:
        """Lazy initialization of Chroma client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            persist_dir = str(self._config.persist_directory or "./data/chroma")
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._config.collection_name or "bhodi"
            )

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with embeddings to Chroma."""
        await self._ensure_client()

        ids = [str(chunk.id) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list:
        """Search for similar chunks."""
        await self._ensure_client()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        from bhodi_platform.domain.entities import RetrievedDocument
        from bhodi_platform.domain.value_objects import ChunkId, DocumentId

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id_str in enumerate(results["ids"][0]):
                # Parse chunk_id format: "document_uuid:chunk_index"
                parts = chunk_id_str.split(":")
                if len(parts) == 2:
                    doc_uuid, chunk_idx = parts
                    doc_id = DocumentId(doc_uuid)
                    chunk_id = ChunkId(document_id=doc_id, chunk_index=int(chunk_idx))
                else:
                    doc_id = DocumentId(chunk_id_str)
                    chunk_id = ChunkId(document_id=doc_id, chunk_index=0)

                retrieved.append(
                    RetrievedDocument(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        text=results["documents"][0][i],
                        score=results["distances"][0][i]
                        if results.get("distances")
                        else None,
                        metadata=results["metadatas"][0][i]
                        if results.get("metadatas")
                        else {},
                    )
                )

        return retrieved

    async def delete(self, document_id: DocumentId) -> None:
        """Delete all chunks for a document."""
        await self._ensure_client()

        # Query to find all chunks for this document
        result = self._collection.get(where={"document_id": str(document_id)})

        if result["ids"]:
            self._collection.delete(ids=result["ids"])

    async def persist(self) -> None:
        """Chroma persists automatically with PersistentClient."""
        pass
