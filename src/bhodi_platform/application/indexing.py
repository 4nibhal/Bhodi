"""Document indexing use case."""

from __future__ import annotations

from bhodi_platform.application.models import (
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from bhodi_platform.ports.chunker import ChunkerPort
from bhodi_platform.ports.document_parser import DocumentParserPort
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.vector_store import VectorStorePort


class IndexDocumentUseCase:
    def __init__(
        self,
        document_parser: DocumentParserPort,
        chunker: ChunkerPort,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._parser = document_parser
        self._chunker = chunker
        self._embedding = embedding
        self._vector_store = vector_store

    async def execute(self, request: IndexDocumentRequest) -> IndexDocumentResponse:
        pass
