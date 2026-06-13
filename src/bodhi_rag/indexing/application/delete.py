"""
DeleteDocumentUseCase.

Removes a document (and all its chunks) from the vector store.
Returns None; the only signal of outcome is whether the adapter
raises (typically DocumentNotFoundError for a missing id).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.domain.value_objects import DocumentId
    from bodhi_rag.ports.vector_store import VectorStorePort


class DeleteDocumentUseCase:
    """Application-layer entry point for deleting a document by id."""

    def __init__(self, *, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    async def execute(self, document_id: DocumentId) -> None:
        """Delete the document identified by `document_id` and all its chunks."""
        await self._vector_store.delete(document_id)
