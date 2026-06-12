"""Unit tests for the dependency injection container."""

from __future__ import annotations

import pytest

from bodhi_rag.application.config import (
    BhodiConfig,
    ChunkerConfig,
    ConversationConfig,
    DocumentParserConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorStoreConfig,
)
from bodhi_rag.infrastructure.container import Container


def _make_config(*, embedding: str = "mock", vector_store: str = "in_memory") -> BhodiConfig:
    return BhodiConfig(
        embedding=EmbeddingConfig(provider=embedding, dimensions=8),
        vector_store=VectorStoreConfig(provider=vector_store),
        chunker=ChunkerConfig(provider="fixed_size", chunk_size=64),
        parser=DocumentParserConfig(provider="mock"),
        llm=LLMConfig(provider="mock"),
        conversation=ConversationConfig(provider="volatile"),
    )


def test_build_raises_for_unknown_embedding_provider():
    """Unknown embedding providers should fail fast."""
    container = Container(_make_config(embedding="mystery"))

    with pytest.raises(ValueError, match="Unknown embedding provider: mystery"):
        container.build()


def test_build_raises_for_unknown_vector_store_provider():
    """Unknown vector store providers should fail fast."""
    container = Container(_make_config(vector_store="mystery"))

    with pytest.raises(ValueError, match="Unknown vector_store provider: mystery"):
        container.build()
