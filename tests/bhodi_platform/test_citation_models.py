"""Tests for citation models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

from bhodi_platform.application.citation_models import DocumentToContext


class DocumentToContextTest(TestCase):
    def test_document_to_context_is_frozen(self) -> None:
        doc = DocumentToContext(original_content="test content")
        with self.assertRaises(AttributeError):
            doc.original_content = "modified"

    def test_from_retrieved_document(self) -> None:
        source_doc = SimpleNamespace(
            page_content="original content",
            metadata={"source": "test.txt", "page": 1},
        )
        doc_to_context = DocumentToContext.from_retrieved_document(source_doc)

        self.assertEqual(doc_to_context.original_content, "original content")
        self.assertEqual(doc_to_context.page_content, "original content")
        self.assertEqual(doc_to_context.metadata["source"], "test.txt")
        self.assertEqual(doc_to_context.metadata["page"], 1)
        self.assertIsNone(doc_to_context.summary)
        self.assertEqual(doc_to_context.source_id, id(source_doc))

    def test_from_retrieved_document_with_custom_source_id(self) -> None:
        source_doc = SimpleNamespace(page_content="content", metadata={})
        doc_to_context = DocumentToContext.from_retrieved_document(
            source_doc, source_id=42
        )
        self.assertEqual(doc_to_context.source_id, 42)

    def test_from_retrieved_document_rejects_non_string_page_content(self) -> None:
        source_doc = SimpleNamespace(page_content=None, metadata={})
        with self.assertRaisesRegex(TypeError, "page_content"):
            DocumentToContext.from_retrieved_document(source_doc)

    def test_with_summary(self) -> None:
        original = DocumentToContext(
            original_content="long content here" * 100,
            metadata={"source": "doc.txt"},
            page_content="long content here" * 100,
        )
        summarized = original.with_summary("short summary")

        # Original should be unchanged
        self.assertEqual(original.original_content, "long content here" * 100)
        self.assertIsNone(original.summary)
        self.assertEqual(original.page_content, "long content here" * 100)

        # New instance should have summary
        self.assertEqual(summarized.original_content, "long content here" * 100)
        self.assertEqual(summarized.summary, "short summary")
        self.assertEqual(summarized.page_content, "short summary")
        self.assertEqual(summarized.metadata["source"], "doc.txt")
        self.assertEqual(summarized.source_id, original.source_id)

    def test_with_page_content(self) -> None:
        original = DocumentToContext(
            original_content="full content",
            metadata={"source": "doc.txt"},
            page_content="full content",
        )
        truncated = original.with_page_content("partial")

        # Original should be unchanged
        self.assertEqual(original.original_content, "full content")
        self.assertEqual(original.page_content, "full content")

        # New instance should have truncated content
        self.assertEqual(truncated.original_content, "full content")
        self.assertEqual(truncated.page_content, "partial")
        self.assertEqual(truncated.metadata["source"], "doc.txt")
        self.assertEqual(truncated.source_id, original.source_id)

    def test_is_summarized(self) -> None:
        unsummarized = DocumentToContext(
            original_content="content", page_content="content"
        )
        summarized = DocumentToContext(
            original_content="long content", page_content="summary", summary="summary"
        )

        self.assertFalse(unsummarized.is_summarized)
        self.assertTrue(summarized.is_summarized)

    def test_is_truncated(self) -> None:
        # Not truncated - content matches original
        regular = DocumentToContext(original_content="content", page_content="content")
        # Truncated - content differs but no summary
        truncated = DocumentToContext(
            original_content="full content", page_content="partial"
        )
        # Not truncated - content differs but has summary
        summarized = DocumentToContext(
            original_content="full content", page_content="summary", summary="summary"
        )

        self.assertFalse(regular.is_truncated)
        self.assertTrue(truncated.is_truncated)
        self.assertFalse(summarized.is_truncated)

    def test_source_id_preserved_through_transformations(self) -> None:
        original = DocumentToContext(
            original_content="content",
            metadata={},
            source_id=12345,
        )
        summarized = original.with_summary("summary")
        truncated = original.with_page_content("partial")

        self.assertEqual(original.source_id, 12345)
        self.assertEqual(summarized.source_id, 12345)
        self.assertEqual(truncated.source_id, 12345)

    def test_metadata_preserved_through_transformations(self) -> None:
        original = DocumentToContext(
            original_content="content",
            metadata={"key": "value", "num": 42},
        )
        summarized = original.with_summary("summary")
        truncated = original.with_page_content("partial")

        self.assertEqual(original.metadata, {"key": "value", "num": 42})
        self.assertEqual(summarized.metadata, {"key": "value", "num": 42})
        self.assertEqual(truncated.metadata, {"key": "value", "num": 42})

    def test_immutability_ensures_no_source_mutation(self) -> None:
        """Verify that creating DocumentToContext doesn't mutate source document."""
        source_doc = SimpleNamespace(
            page_content="original content",
            metadata={"source": "test.txt"},
        )
        original_content = source_doc.page_content
        original_metadata = dict(source_doc.metadata)

        doc_to_context = DocumentToContext.from_retrieved_document(source_doc)

        # Modify the doc_to_context
        summarized = doc_to_context.with_summary("summary")

        # Source document should be unchanged
        self.assertEqual(source_doc.page_content, original_content)
        self.assertEqual(source_doc.metadata, original_metadata)

    def test_slots(self) -> None:
        doc = DocumentToContext(original_content="test")
        self.assertTrue(hasattr(doc, "__slots__"))
