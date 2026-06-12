"""
Mock document parser adapter for testing.

Returns simple documents without actual parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from bodhi_rag.application.config import DocumentParserConfig
    from bodhi_rag.domain.entities import Document


class MockDocumentParserAdapter:
    """
    Mock document parser for testing.

    Returns a simple document with placeholder content.
    """

    def __init__(self, config: DocumentParserConfig) -> None:
        self._config = config

    async def parse(
        self,
        source: Path | bytes | BinaryIO,
    ) -> Document:
        """Return a mock document."""
        from bodhi_rag.domain.entities import Document
        from bodhi_rag.domain.value_objects import DocumentId

        # Determine source name
        if isinstance(source, Path):
            source_name = str(source)
        elif isinstance(source, str):
            source_name = source
        else:
            source_name = "unknown"

        return Document(
            id=DocumentId(),
            text=f"Content from {source_name}. This is mock content for testing.",
            metadata={"source": source_name},
        )

    async def extract_text(
        self,
        _source: Path | bytes | BinaryIO,
    ) -> str:
        """Return mock text."""
        return "Mock extracted text content."

    async def extract_metadata(
        self,
        source: Path | bytes | BinaryIO,
    ) -> dict:
        """Return mock metadata."""
        if isinstance(source, Path):
            return {"source": str(source), "type": "mock"}
        return {"type": "mock"}
