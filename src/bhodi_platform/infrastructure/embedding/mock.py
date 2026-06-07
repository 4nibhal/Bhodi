"""
Mock embedding adapter for testing.

Provides deterministic fake embeddings without network calls.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import EmbeddingConfig


class MockEmbeddingAdapter:
    """
    Fake embedding adapter that returns deterministic embeddings.

    Useful for E2E tests that don't want external API calls.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._dimensions = config.dimensions or 384

    @traced("mock.embed_documents")
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return fake embeddings based on a deterministic text digest."""
        return [self._fake_embedding(t) for t in texts]

    @traced("mock.embed_query")
    async def embed_query(self, text: str) -> list[float]:
        """Return fake query embedding."""
        return self._fake_embedding(text)

    async def dimensions(self) -> int:
        """Return configured dimensions."""
        return self._dimensions

    def _fake_embedding(self, text: str) -> list[float]:
        """Generate deterministic fake embedding from text."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        bytes_needed = self._dimensions * 2
        random_bytes = bytearray()
        counter = 0

        while len(random_bytes) < bytes_needed:
            counter_bytes = counter.to_bytes(4, byteorder="big", signed=False)
            random_bytes.extend(hashlib.sha256(seed + counter_bytes).digest())
            counter += 1

        embedding: list[float] = []
        for index in range(0, bytes_needed, 2):
            chunk = random_bytes[index : index + 2]
            value = int.from_bytes(chunk, byteorder="big", signed=False)
            normalized = value / 65535.0
            embedding.append((normalized * 2.0) - 1.0)

        return embedding
