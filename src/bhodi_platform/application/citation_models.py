"""Citation models for Bhodi platform.

These models track how documents are prepared for context assembly
without mutating the original LangChain document objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentToContext:
    """Immutable representation of a document prepared for context assembly.

    This class wraps LangChain document data and tracks any transformations
    (summarization, page_content modification) without mutating the original.

    Attributes:
        original_content: The original page_content from the LangChain document.
        metadata: Metadata from the original document.
        summary: Optional summary if the document was summarized.
        page_content: The content to use in context (original or summarized).
        source_id: Identity marker for the source document (typically id(document)).
    """

    original_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    page_content: str = ""
    source_id: int | None = None

    def __post_init__(self) -> None:
        # Ensure page_content defaults to original_content if not explicitly set
        if not self.page_content:
            object.__setattr__(self, "page_content", self.original_content)

    @classmethod
    def from_retrieved_document(
        cls, document: Any, source_id: int | None = None
    ) -> "DocumentToContext":
        """Create a DocumentToContext from a LangChain document.

        This factory extracts data without mutating the source document.

        Args:
            document: A LangChain document-like object with page_content and metadata.
            source_id: Optional identity marker for the source document.

        Returns:
            A new DocumentToContext instance.

        Raises:
            TypeError: If document does not have string page_content.
        """
        page_content = cls._extract_page_content(document)
        metadata = cls._extract_metadata(document)
        return cls(
            original_content=page_content,
            metadata=metadata,
            page_content=page_content,
            source_id=source_id if source_id is not None else id(document),
        )

    def with_summary(self, summary: str) -> "DocumentToContext":
        """Create a new DocumentToContext with a summary applied.

        The page_content will be the summary, but original_content is preserved.

        Args:
            summary: The summarized text.

        Returns:
            A new DocumentToContext with summary set.
        """
        return DocumentToContext(
            original_content=self.original_content,
            metadata=self.metadata,
            summary=summary,
            page_content=summary,
            source_id=self.source_id,
        )

    def with_page_content(self, content: str) -> "DocumentToContext":
        """Create a new DocumentToContext with different page_content.

        This is useful when partial content is used due to token budget.
        The original_content is preserved for diagnostics.

        Args:
            content: The new page_content.

        Returns:
            A new DocumentToContext with updated page_content.
        """
        return DocumentToContext(
            original_content=self.original_content,
            metadata=self.metadata,
            summary=self.summary,
            page_content=content,
            source_id=self.source_id,
        )

    @property
    def is_summarized(self) -> bool:
        """Check if this document has been summarized."""
        return self.summary is not None

    @property
    def is_truncated(self) -> bool:
        """Check if this document's page_content differs from original_content.

        Note: This is different from is_summarized. A document could be truncated
        (partial content due to token budget) without being summarized.
        """
        return self.page_content != self.original_content and self.summary is None

    @staticmethod
    def _extract_page_content(document: Any) -> str:
        """Extract and validate page_content from a document."""
        page_content = getattr(document, "page_content", None)
        if not isinstance(page_content, str):
            raise TypeError(
                "Retrieved documents must expose string page_content to preserve "
                "content and metadata for future citation support."
            )
        return page_content

    @staticmethod
    def _extract_metadata(document: Any) -> dict[str, Any]:
        """Extract and normalize metadata from a document."""
        raw_metadata = getattr(document, "metadata", {})
        if not isinstance(raw_metadata, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key, value in raw_metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[str(key)] = value
            else:
                normalized[str(key)] = str(value)
        return normalized
