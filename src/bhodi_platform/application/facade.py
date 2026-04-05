"""Bhodi application facade - orchestrates use cases via protocol ports."""

from __future__ import annotations

from bhodi_platform.application.models import (
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
    CitationResponse,
)
from bhodi_platform.ports.chunker import ChunkerPort
from bhodi_platform.ports.conversation_memory import ConversationMemoryPort
from bhodi_platform.ports.document_parser import DocumentParserPort
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.llm import LLMPort
from bhodi_platform.ports.vector_store import VectorStorePort


class BhodiApplication:
    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        chunker: ChunkerPort,
        document_parser: DocumentParserPort,
        llm: LLMPort,
        conversation_memory: ConversationMemoryPort,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._chunker = chunker
        self._document_parser = document_parser
        self._llm = llm
        self._conversation_memory = conversation_memory

    async def index_document(
        self, request: IndexDocumentRequest
    ) -> IndexDocumentResponse:
        doc = await self._document_parser.parse(request.source)

        chunks = await self._chunker.chunk(
            doc.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )

        embeddings = await self._embedding.embed_documents([c.content for c in chunks])

        await self._vector_store.add(chunks, embeddings)

        return IndexDocumentResponse(
            document_id=str(doc.id),
            chunk_count=len(chunks),
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = await self._embedding.embed_query(request.question)

        retrieved = await self._vector_store.search(query_embedding, request.top_k)

        answer_text = await self._llm.generate_with_context(
            request.question,
            retrieved,
            temperature=request.temperature,
        )

        citations = [
            CitationResponse(
                chunk_id=str(doc.chunk_id),
                text=doc.text[:200],
                source_document=str(doc.document_id),
            )
            for doc in retrieved
        ]

        return QueryResponse(
            answer_text=answer_text,
            citations=citations,
            conversation_id=request.conversation_id,
        )

    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="healthy", version="1.0.0")

    async def delete_document(self, document_id) -> None:
        """Delete a document and all its chunks."""
        await self._vector_store.delete(document_id)

    async def get_conversation_history(self, conversation_id, limit: int | None = None):
        """Get conversation history for a given conversation ID."""
        return await self._conversation_memory.get_history(conversation_id, limit)
