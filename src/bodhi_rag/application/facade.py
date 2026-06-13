"""bodhi-rag application facade - orchestrates use cases via protocol ports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bodhi_rag._version import get_version
from bodhi_rag.application.models import (
    CitationResponse,
    HealthStatus,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
)
from bodhi_rag.domain.entities import Chunk, ConversationTurn, Document, RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, ConversationId, DocumentId

if TYPE_CHECKING:
    from bodhi_rag.ports.chunker import ChunkerPort
    from bodhi_rag.ports.conversation_memory import ConversationMemoryPort
    from bodhi_rag.ports.document_parser import DocumentParserPort
    from bodhi_rag.ports.embedding import EmbeddingPort
    from bodhi_rag.ports.llm import LLMPort
    from bodhi_rag.ports.reranker import RerankerPort
    from bodhi_rag.ports.vector_store import VectorStorePort


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
    },
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
            ),
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
        reranker: RerankerPort,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._chunker = chunker
        self._document_parser = document_parser
        self._llm = llm
        self._conversation_memory = conversation_memory
        self._reranker = reranker

    async def index_document(
        self, request: IndexDocumentRequest,
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
            [chunk.content for chunk in rebound_chunks],
        )

        await self._vector_store.add(rebound_chunks, embeddings)

        return IndexDocumentResponse(
            document_id=str(document.id),
            chunk_count=len(rebound_chunks),
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = await self._embedding.embed_query(request.question)

        # Overfetch: ask the vector store for more candidates than the final
        # top_k so the reranker has room to reorder without dropping
        # high-relevance items that the raw vector similarity ranked just
        # below the cutoff. The factor comes from the active reranker
        # adapter; NoOpReranker typically reports 1 (no overfetch).
        overfetch_top_k = max(request.top_k * self._reranker.overfetch_factor, request.top_k)

        candidates = await self._vector_store.search(query_embedding, overfetch_top_k)

        reranked = await self._reranker.rerank(
            request.question,
            candidates,
            top_k=request.top_k,
        )

        answer_text = await self._llm.generate_with_context(
            request.question,
            reranked,
            temperature=request.temperature,
        )

        citations = [
            CitationResponse(
                chunk_id=str(doc.chunk_id),
                text=doc.text[:200],
                source_document=_citation_source_document(doc),
                page=_citation_page(doc.metadata),
            )
            for doc in reranked
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

    async def delete_document(self, document_id: DocumentId) -> None:
        """Delete a document and all its chunks."""
        await self._vector_store.delete(document_id)

    async def get_conversation_history(
        self,
        conversation_id: ConversationId,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation history for a given conversation ID."""
        return await self._conversation_memory.get_history(conversation_id, limit)
