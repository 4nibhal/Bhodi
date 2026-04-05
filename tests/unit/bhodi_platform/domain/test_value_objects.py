"""Tests for domain value objects."""

import pytest
from bhodi_platform.domain.value_objects import (
    DocumentId,
    ChunkId,
    ConversationId,
    EmbeddingVector,
    Citation,
)


class TestDocumentId:
    def test_create_with_none_generates_uuid(self):
        doc_id = DocumentId()
        assert doc_id.value is not None

    def test_create_from_string(self):
        uuid_str = "12345678-1234-5678-1234-567812345678"
        doc_id = DocumentId(uuid_str)
        assert str(doc_id) == uuid_str

    def test_equality(self):
        id1 = DocumentId("12345678-1234-5678-1234-567812345678")
        id2 = DocumentId("12345678-1234-5678-1234-567812345678")
        assert id1 == id2


class TestChunkId:
    def test_create_chunk_id(self):
        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        assert chunk_id.chunk_index == 0

    def test_chunk_id_validates_negative_index(self):
        doc_id = DocumentId()
        with pytest.raises(ValueError):
            ChunkId(document_id=doc_id, chunk_index=-1)

    def test_chunk_id_string_representation(self):
        doc_id = DocumentId("12345678-1234-5678-1234-567812345678")
        chunk_id = ChunkId(document_id=doc_id, chunk_index=2)
        assert "12345678" in str(chunk_id)
        assert ":2" in str(chunk_id)


class TestEmbeddingVector:
    def test_create_from_list(self):
        vec = EmbeddingVector([0.1, 0.2, 0.3])
        assert len(vec) == 3
        assert vec[0] == 0.1

    def test_equality(self):
        vec1 = EmbeddingVector([1.0, 2.0])
        vec2 = EmbeddingVector([1.0, 2.0])
        assert vec1 == vec2

    def test_immutability(self):
        vec = EmbeddingVector([1.0, 2.0])
        with pytest.raises(Exception):  # frozen dataclass
            vec.values[0] = 0.0
