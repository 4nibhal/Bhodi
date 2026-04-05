"""Tests for domain entities."""

from __future__ import annotations

from unittest import TestCase

from bhodi_platform.domain import (
    Answer,
    Chunk,
    ConversationTurn,
    IndexedDocument,
    Query,
    RetrievedDocument,
)
from bhodi_platform.domain.value_objects import (
    ChunkId,
    Citation,
    DocumentId,
    ConversationId,
)


class QueryEntityTest(TestCase):
    def test_query_is_frozen(self) -> None:
        query = Query(text="test query")
        with self.assertRaises(AttributeError):
            query.text = "modified"

    def test_query_requires_non_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            Query(text="")

    def test_query_has_conversation_id(self) -> None:
        conv_id = ConversationId()
        query = Query(text="test", conversation_id=conv_id)
        self.assertEqual(query.conversation_id, conv_id)

    def test_query_slots(self) -> None:
        query = Query(text="test")
        self.assertTrue(hasattr(query, "__slots__"))


class RetrievedDocumentEntityTest(TestCase):
    def test_retrieved_document_is_frozen(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        retrieved = RetrievedDocument(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="content",
        )
        with self.assertRaises(AttributeError):
            retrieved.text = "modified"

    def test_retrieved_document_properties(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        retrieved = RetrievedDocument(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="content",
            score=0.95,
            metadata={"source": "test"},
        )
        self.assertEqual(retrieved.text, "content")
        self.assertEqual(retrieved.metadata["source"], "test")
        self.assertEqual(retrieved.score, 0.95)

    def test_retrieved_document_slots(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        retrieved = RetrievedDocument(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="content",
        )
        self.assertTrue(hasattr(retrieved, "__slots__"))


class AnswerEntityTest(TestCase):
    def test_answer_is_frozen(self) -> None:
        answer = Answer(text="The answer is 42")
        with self.assertRaises(AttributeError):
            answer.text = "modified"

    def test_answer_with_confidence(self) -> None:
        answer = Answer(text="Answer", confidence=0.95)
        self.assertEqual(answer.confidence, 0.95)

    def test_answer_citations(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        citation = Citation(
            chunk_id=chunk_id,
            text="source text",
            source_document="doc.pdf",
            page=1,
        )
        answer = Answer(text="Answer", citations=(citation,))
        self.assertEqual(answer.citations, (citation,))

    def test_answer_slots(self) -> None:
        answer = Answer(text="test")
        self.assertTrue(hasattr(answer, "__slots__"))


class ConversationTurnEntityTest(TestCase):
    def test_conversation_turn_is_frozen(self) -> None:
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hi",
            assistant_message="Hello",
        )
        with self.assertRaises(AttributeError):
            turn.user_message = "modified"

    def test_conversation_turn_fields(self) -> None:
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hi",
            assistant_message="Hello",
            turn_index=5,
        )
        self.assertEqual(turn.user_message, "Hi")
        self.assertEqual(turn.assistant_message, "Hello")
        self.assertEqual(str(turn.conversation_id), str(conv_id))
        self.assertEqual(turn.turn_index, 5)

    def test_conversation_turn_slots(self) -> None:
        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hi",
            assistant_message="Hello",
        )
        self.assertTrue(hasattr(turn, "__slots__"))


class ChunkEntityTest(TestCase):
    def test_chunk_is_frozen(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="test",
            chunk_index=0,
            total_chunks=1,
        )
        with self.assertRaises(AttributeError):
            chunk.content = "modified"

    def test_chunk_fields(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 2)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="test content",
            chunk_index=2,
            total_chunks=5,
            metadata={"page": 3},
        )
        self.assertEqual(chunk.content, "test content")
        self.assertEqual(chunk.chunk_index, 2)
        self.assertEqual(chunk.total_chunks, 5)
        self.assertEqual(chunk.metadata["page"], 3)

    def test_chunk_slots(self) -> None:
        doc_id = DocumentId()
        chunk_id = ChunkId(doc_id, 0)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="test",
            chunk_index=0,
            total_chunks=1,
        )
        self.assertTrue(hasattr(chunk, "__slots__"))


class IndexedDocumentEntityTest(TestCase):
    def test_indexed_document_is_frozen(self) -> None:
        doc = IndexedDocument(
            source="doc.md",
            chunk_count=3,
            indexed_at="2024-01-01T00:00:00",
        )
        with self.assertRaises(AttributeError):
            doc.source = "modified"

    def test_indexed_document_with_chunks(self) -> None:
        doc_id = DocumentId()
        chunk1 = Chunk(
            id=ChunkId(doc_id, 0),
            document_id=doc_id,
            content="part1",
            chunk_index=0,
            total_chunks=2,
        )
        chunk2 = Chunk(
            id=ChunkId(doc_id, 1),
            document_id=doc_id,
            content="part2",
            chunk_index=1,
            total_chunks=2,
        )
        doc = IndexedDocument(
            source="doc.md",
            chunk_count=2,
            indexed_at="2024-01-01T00:00:00",
            chunks=(chunk1, chunk2),
        )
        self.assertEqual(len(doc.chunks), 2)
        self.assertEqual(doc.chunks[0].content, "part1")

    def test_indexed_document_slots(self) -> None:
        doc = IndexedDocument(
            source="doc.md",
            chunk_count=1,
            indexed_at="2024-01-01T00:00:00",
        )
        self.assertTrue(hasattr(doc, "__slots__"))
