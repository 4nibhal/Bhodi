"""
Mock embedding adapter for testing.

Provides deterministic fake embeddings without network calls.
"""

from __future__ import annotations

import random
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
        """Return fake embeddings based on text hash."""
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
        # Use text hash for reproducible "random" values
        rng = random.Random(hash(text) % (2**32))
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]
