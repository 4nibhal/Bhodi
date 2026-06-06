"""
PyPDF document parser adapter.

Parses PDF documents using pypdf library.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO, TYPE_CHECKING

from bhodi_platform.domain.entities import Document
from bhodi_platform.domain.value_objects import DocumentId
from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import DocumentParserConfig


TEXT_FILE_SUFFIXES = frozenset({".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"})


class PyPDFDocumentParserAdapter:
    """
    PyPDF-based document parser.

    Extracts text and metadata from PDF files.
    """

    def __init__(self, config: DocumentParserConfig) -> None:
        self._config = config

    def _build_path_metadata(self, path: Path, file_type: str) -> dict[str, Any]:
        return {
            "source": path.name,
            "source_path": str(path.resolve()),
            "filename": path.name,
            "file_type": file_type,
        }

    def _extract_pdf_contents(self, reader: Any, metadata: dict[str, Any]) -> str:
        text_parts: list[str] = []
        metadata["page_count"] = len(reader.pages)

        if reader.metadata:
            metadata["author"] = reader.metadata.get("/Author", "")
            metadata["title"] = reader.metadata.get("/Title", "")
            metadata["subject"] = reader.metadata.get("/Subject", "")

        for page_num, page in enumerate(reader.pages, 1):
            text_parts.append(f"--- Page {page_num} ---\n{page.extract_text() or ''}")

        return "\n".join(text_parts)

    def _parse_text_path_sync(self, path: Path) -> Document:
        metadata = self._build_path_metadata(path, file_type="text")
        with open(path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()

        return Document(
            id=DocumentId(),
            text=text,
            metadata=metadata,
        )

    def _parse_pdf_path_sync(self, path: Path) -> Document:
        import pypdf

        metadata = self._build_path_metadata(path, file_type="pdf")
        with open(path, "rb") as file_handle:
            reader = pypdf.PdfReader(file_handle)
            full_text = self._extract_pdf_contents(reader, metadata)

        return Document(
            id=DocumentId(),
            text=full_text,
            metadata=metadata,
        )

    def _parse_bytes_sync(self, source: bytes) -> Document:
        import io
        import pypdf

        metadata: dict[str, Any] = {
            "source": "bytes",
            "byte_length": len(source),
            "file_type": "pdf",
        }
        reader = pypdf.PdfReader(io.BytesIO(source))
        full_text = self._extract_pdf_contents(reader, metadata)

        return Document(
            id=DocumentId(),
            text=full_text,
            metadata=metadata,
        )

    def _parse_stream_sync(self, source: BinaryIO) -> Document:
        import pypdf

        metadata: dict[str, Any] = {"source": "stream", "file_type": "pdf"}
        source_name = getattr(source, "name", None)
        if source_name:
            metadata["filename"] = Path(str(source_name)).name

        reader = pypdf.PdfReader(source)
        full_text = self._extract_pdf_contents(reader, metadata)

        return Document(
            id=DocumentId(),
            text=full_text,
            metadata=metadata,
        )

    @traced("pypdf.parse")
    async def parse(
        self,
        source: str | Path | bytes | BinaryIO,
    ) -> Document:
        """Parse a PDF document and return a Document entity."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in TEXT_FILE_SUFFIXES:
                return await asyncio.to_thread(self._parse_text_path_sync, path)
            return await asyncio.to_thread(self._parse_pdf_path_sync, path)

        if isinstance(source, bytes):
            return await asyncio.to_thread(self._parse_bytes_sync, source)

        if hasattr(source, "read"):
            return await asyncio.to_thread(self._parse_stream_sync, source)

        raise ValueError(f"Unsupported source type: {type(source)}")

    @traced("pypdf.extract_text")
    async def extract_text(
        self,
        source: Path | bytes | BinaryIO,
    ) -> str:
        """Extract raw text from a PDF document."""
        doc = await self.parse(source)
        return doc.text

    @traced("pypdf.extract_metadata")
    async def extract_metadata(
        self,
        source: Path | bytes | BinaryIO,
    ) -> dict:
        """Extract metadata from a PDF document."""
        doc = await self.parse(source)
        return doc.metadata
