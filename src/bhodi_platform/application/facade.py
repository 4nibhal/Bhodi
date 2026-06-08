"""Bhodi application facade - orchestrates use cases via protocol ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bhodi_platform.application.models import (
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
    CitationResponse,
)
from bhodi_platform._version import get_version
from bhodi_platform.domain.entities import Chunk, Document, RetrievedDocument
from bhodi_platform.domain.value_objects import ChunkId
from bhodi_platform.ports.chunker import ChunkerPort
from bhodi_platform.ports.conversation_memory import ConversationMemoryPort
from bhodi_platform.ports.document_parser import DocumentParserPort
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.llm import LLMPort
from bhodi_platform.ports.vector_store import VectorStorePort


RESERVED_PROVENANCE_KEYS = frozenset(
    {
        "source",
        "source_path",
        "filename",
        "file_type",
        "page_count",
        "author",
        "title",
        "subject",
    }
)


def _merge_document_metadata(
    parser_metadata: dict[str, Any],
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(parser_metadata)
    for key, value in request_metadata.items():
        if key in RESERVED_PROVENANCE_KEYS:
            merged[f"user_{key}"] = value
            continue
        merged[key] = value
    return merged


def _rebind_chunks(document: Document, chunks: list[Chunk]) -> list[Chunk]:
    total_chunks = len(chunks)
    rebound_chunks: list[Chunk] = []

    for index, chunk in enumerate(chunks):
        rebound_chunks.append(
            Chunk(
                id=ChunkId(document_id=document.id, chunk_index=index),
                document_id=document.id,
                content=chunk.content,
                chunk_index=index,
                total_chunks=total_chunks,
                metadata={**document.metadata, **chunk.metadata},
            )
        )

    return rebound_chunks


def _citation_source_document(retrieved_document: RetrievedDocument) -> str:
    filename = retrieved_document.metadata.get("filename")
    if isinstance(filename, str) and filename:
        return filename

    source = retrieved_document.metadata.get("source")
    if isinstance(source, str) and source and source not in {"bytes", "stream"}:
        source_name = Path(source).name
        if source_name:
            return source_name

    return str(retrieved_document.document_id)


def _citation_page(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page")
    if page is None or isinstance(page, bool):
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


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
        parsed_document = await self._document_parser.parse(request.source)
        document = Document(
            id=parsed_document.id,
            text=parsed_document.text,
            metadata=_merge_document_metadata(
                parsed_document.metadata,
                request.metadata,
            ),
            indexed_at=parsed_document.indexed_at,
        )

        chunks = await self._chunker.chunk(
            document.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )

        rebound_chunks = _rebind_chunks(document, chunks)

        embeddings = await self._embedding.embed_documents(
            [chunk.content for chunk in rebound_chunks]
        )

        await self._vector_store.add(rebound_chunks, embeddings)

        return IndexDocumentResponse(
            document_id=str(document.id),
            chunk_count=len(rebound_chunks),
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
                source_document=_citation_source_document(doc),
                page=_citation_page(doc.metadata),
            )
            for doc in retrieved
        ]

        return QueryResponse(
            answer_text=answer_text,
            citations=citations,
            conversation_id=request.conversation_id,
        )

    async def health_check(self) -> HealthStatus:
        services = {
            "embedding": self._embedding is not None,
            "vector_store": self._vector_store is not None,
            "llm": self._llm is not None,
        }

        status = "healthy" if all(services.values()) else "degraded"
        return HealthStatus(
            status=status,
            version=get_version(),
            services=services,
        )

    async def delete_document(self, document_id) -> None:
        """Delete a document and all its chunks."""
        await self._vector_store.delete(document_id)

    async def get_conversation_history(self, conversation_id, limit: int | None = None):
        """Get conversation history for a given conversation ID."""
        return await self._conversation_memory.get_history(conversation_id, limit)
