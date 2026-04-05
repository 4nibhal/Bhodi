"""
Fixed-size chunker adapter.

Splits text into chunks of fixed size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bhodi_platform.domain.entities import Chunk
from bhodi_platform.domain.value_objects import ChunkId, DocumentId
from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import ChunkerConfig


class FixedSizeChunkerAdapter:
    """
    Fixed-size text chunker.

    Splits text into chunks of approximately chunk_size characters.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_OVERLAP = 64

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config
        self._chunk_size = config.chunk_size or self.DEFAULT_CHUNK_SIZE
        self._overlap = config.overlap if config.overlap is not None else self.DEFAULT_OVERLAP

    @traced("chunker.fixed_size.chunk")
    async def chunk(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[Chunk]:
        """Split text into fixed-size chunks."""
        size = chunk_size or self._chunk_size
        ov = overlap if overlap is not None else self._overlap

        if size <= 0:
            raise ValueError("chunk_size must be positive")
        if ov < 0:
            raise ValueError("overlap must be non-negative")
        if ov >= size:
            raise ValueError("overlap must be less than chunk_size")

        # Collect raw chunks
        raw_chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + size, text_len)

            # Don't cut in the middle of a word if possible
            if end < text_len and text[end - 1] != " ":
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                raw_chunks.append(chunk_text)

            # Move start position for next chunk
            # If we're at the end, break to avoid infinite loop
            if end >= text_len:
                break

            # Advance by size - overlap
            next_start = end - ov
            if next_start <= start:
                # Overlap too large or zero size advance, force move by size
                next_start = start + size

            if next_start >= text_len:
                # Final position would be beyond text, append final chunk if different
                break

            start = next_start

        # Create proper Chunk objects
        total_chunks = len(raw_chunks)
        if total_chunks == 0:
            return []

        doc_id = DocumentId()
        chunks: list[Chunk] = []

        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = ChunkId(document_id=doc_id, chunk_index=index)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content=chunk_text,
                    chunk_index=index,
                    total_chunks=total_chunks,
                )
            )

        return chunks

    @property
    def default_chunk_size(self) -> int:
        """Return the default chunk size."""
        return self._chunk_size

    @property
    def default_overlap(self) -> int:
        """Return the default overlap."""
        return self._overlap
