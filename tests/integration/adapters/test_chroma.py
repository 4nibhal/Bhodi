"""
Integration tests for Chroma vector store adapter.

Requires chromadb package to be installed.
"""

import pytest
import tempfile
from pathlib import Path

from bhodi_platform.application.config import VectorStoreConfig
from bhodi_platform.infrastructure.vector_store.chroma import ChromaVectorStoreAdapter
from bhodi_platform.domain.entities import Chunk
from bhodi_platform.domain.value_objects import ChunkId, DocumentId


@pytest.fixture
def temp_dir():
    """Create a temporary directory for Chroma persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def chroma_adapter(temp_dir):
    """Create a Chroma adapter with temporary directory."""
    config = VectorStoreConfig(
        provider="chroma",
        persist_directory=temp_dir,
        collection_name="test_collection",
    )
    return ChromaVectorStoreAdapter(config)


@pytest.mark.asyncio
async def test_add_and_search(chroma_adapter):
    """Test adding chunks and searching."""
    doc_id = DocumentId()
    chunk_id = ChunkId(document_id=doc_id, chunk_index=0)

    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        content="This is a test document about machine learning.",
        chunk_index=0,
        total_chunks=1,
    )

    embedding = [0.1] * 128  # Mock embedding

    await chroma_adapter.add([chunk], [embedding])

    results = await chroma_adapter.search(embedding, top_k=1)

    assert len(results) == 1
    assert results[0].text == "This is a test document about machine learning."
    assert results[0].document_id == doc_id


@pytest.mark.asyncio
async def test_delete(chroma_adapter):
    """Test deleting a document."""
    doc_id = DocumentId()
    chunk_id = ChunkId(document_id=doc_id, chunk_index=0)

    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        content="Document to delete.",
        chunk_index=0,
        total_chunks=1,
    )

    embedding = [0.1] * 128

    await chroma_adapter.add([chunk], [embedding])

    # Verify it exists
    results = await chroma_adapter.search(embedding, top_k=1)
    assert len(results) == 1

    # Delete
    await chroma_adapter.delete(doc_id)

    # Should not find it anymore
    results = await chroma_adapter.search(embedding, top_k=1)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_persist(chroma_adapter):
    """Test that persist doesn't error."""
    # Persist should be a no-op for Chroma with PersistentClient
    await chroma_adapter.persist()


@pytest.mark.asyncio
async def test_multiple_chunks(chroma_adapter):
    """Test adding and searching multiple chunks."""
    doc_id = DocumentId()

    chunks = []
    embeddings = []
    for i in range(3):
        chunk_id = ChunkId(document_id=doc_id, chunk_index=i)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content=f"Chunk {i} content",
            chunk_index=i,
            total_chunks=3,
        )
        chunks.append(chunk)
        embeddings.append([0.1 + i * 0.01] * 128)

    await chroma_adapter.add(chunks, embeddings)

    results = await chroma_adapter.search(embeddings[0], top_k=3)

    assert len(results) == 3
