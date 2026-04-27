"""
PyPDF document parser adapter.

Parses PDF documents using pypdf library.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TYPE_CHECKING

from bhodi_platform.domain.entities import Document
from bhodi_platform.domain.value_objects import DocumentId
from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import DocumentParserConfig


class PyPDFDocumentParserAdapter:
    """
    PyPDF-based document parser.

    Extracts text and metadata from PDF files.
    """

    def __init__(self, config: DocumentParserConfig) -> None:
        self._config = config

    @traced("pypdf.parse")
    async def parse(
        self,
        source: Path | bytes | BinaryIO,
    ) -> Document:
        """Parse a PDF document and return a Document entity."""
        import pypdf

        text_parts = []
        metadata = {"source": str(source)}

        if isinstance(source, (str, Path)):
            path = Path(source)
            metadata["source"] = str(path.resolve())
            metadata["filename"] = path.name

            # Handle plain text files directly
            if path.suffix.lower() in (".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                metadata["file_type"] = "text"
                return Document(
                    id=DocumentId(),
                    text=text,
                    metadata=metadata,
                )

            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                metadata["page_count"] = len(reader.pages)

                if reader.metadata:
                    metadata["author"] = reader.metadata.get("/Author", "")
                    metadata["title"] = reader.metadata.get("/Title", "")
                    metadata["subject"] = reader.metadata.get("/Subject", "")

                for page_num, page in enumerate(reader.pages, 1):
                    text_parts.append(f"--- Page {page_num} ---\n{page.extract_text()}")

        elif isinstance(source, bytes):
            import io

            reader = pypdf.PdfReader(io.BytesIO(source))
            metadata["page_count"] = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                text_parts.append(f"--- Page {page_num} ---\n{page.extract_text()}")

        elif hasattr(source, "read"):
            # File-like object
            import io

            reader = pypdf.PdfReader(source)
            metadata["page_count"] = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                text_parts.append(f"--- Page {page_num} ---\n{page.extract_text()}")

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        full_text = "\n".join(text_parts)

        return Document(
            id=DocumentId(),
            text=full_text,
            metadata=metadata,
        )

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
