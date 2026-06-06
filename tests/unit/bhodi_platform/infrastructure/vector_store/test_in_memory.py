"""Unit tests for the in-memory vector store adapter."""

from __future__ import annotations

import pytest

from bhodi_platform.application.config import VectorStoreConfig
from bhodi_platform.domain.exceptions import DocumentNotFoundError
from bhodi_platform.domain.value_objects import DocumentId
from bhodi_platform.infrastructure.vector_store.in_memory import MockVectorStoreAdapter


@pytest.mark.asyncio
async def test_delete_raises_not_found_for_missing_document():
    """Deleting a missing document should raise DocumentNotFoundError."""
    adapter = MockVectorStoreAdapter(VectorStoreConfig(provider="in_memory"))

    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        await adapter.delete(DocumentId())
