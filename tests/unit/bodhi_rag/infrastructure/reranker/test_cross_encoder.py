"""
Unit tests for the CrossEncoderReranker adapter.

The optional `cross-encoder` extra (sentence-transformers) is NOT
required to run these tests: we patch the adapter's `_ensure_encoder`
to return a small fake encoder with a `predict(pairs, ...)` method.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from bodhi_rag.application.config import ConfigError, RerankerConfig
from bodhi_rag.domain.entities import RetrievedDocument
from bodhi_rag.domain.value_objects import ChunkId, DocumentId
from bodhi_rag.infrastructure.reranker.cross_encoder import CrossEncoderReranker


def _make_chunk(idx: int, text: str = "default") -> RetrievedDocument:
    doc_id = DocumentId()
    return RetrievedDocument(
        chunk_id=ChunkId(document_id=doc_id, chunk_index=idx),
        document_id=doc_id,
        text=text,
        score=0.0,
        metadata={"idx": idx},
    )


def _fake_encoder(scores: list[float]) -> Any:
    """Build a fake encoder whose `predict` returns the given score list."""
    encoder = MagicMock()
    encoder.predict.return_value = list(scores)
    return encoder


def test_constructor_rejects_empty_model_name() -> None:
    """
    Defensive check: even if config validation is bypassed, the adapter fails fast.

    Pydantic's `model_construct` skips validation so we can construct a
    `RerankerConfig` with an empty model and verify the adapter's own
    defensive guard fires before any SDK is touched.
    """
    config = RerankerConfig.model_construct(provider="cross_encoder", model="")
    with pytest.raises(ConfigError, match="non-empty model name"):
        CrossEncoderReranker(config)


def test_constructor_strips_whitespace_model_name() -> None:
    """Constructor stores the stripped model name and does not eagerly load."""
    config = RerankerConfig(provider="cross_encoder", model="  my-model  ")
    reranker = CrossEncoderReranker(config)
    assert reranker._model_name == "my-model"  # noqa: SLF001  # test introspection
    assert reranker._encoder is None  # noqa: SLF001  # lazy init


@pytest.mark.asyncio
async def test_rerank_reorders_chunks_by_descending_score() -> None:
    """The adapter must reorder chunks by cross-encoder score, highest first."""
    reranker = CrossEncoderReranker(RerankerConfig(provider="cross_encoder", model="m"))
    reranker._encoder = _fake_encoder([0.1, 0.9, 0.5])  # noqa: SLF001  # bypass lazy load

    chunks = [
        _make_chunk(0, text="alpha"),
        _make_chunk(1, text="beta"),
        _make_chunk(2, text="gamma"),
    ]
    result = await reranker.rerank("query", chunks)

    # The order should be the chunk whose pair scored 0.9, then 0.5, then 0.1.
    assert [c.text for c in result] == ["beta", "gamma", "alpha"]
    assert [c.score for c in result] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_rerank_respects_top_k() -> None:
    """When top_k is set, only the first N reranked chunks are returned."""
    reranker = CrossEncoderReranker(RerankerConfig(provider="cross_encoder", model="m"))
    reranker._encoder = _fake_encoder([0.1, 0.9, 0.5, 0.7])  # noqa: SLF001

    chunks = [_make_chunk(i, text=f"t{i}") for i in range(4)]
    result = await reranker.rerank("query", chunks, top_k=2)

    assert len(result) == 2
    # Sorted descending: 0.9 then 0.7
    assert [c.text for c in result] == ["t1", "t3"]


@pytest.mark.asyncio
async def test_rerank_returns_empty_for_empty_input() -> None:
    """Empty input must short-circuit before invoking the encoder."""
    reranker = CrossEncoderReranker(RerankerConfig(provider="cross_encoder", model="m"))
    reranker._encoder = _fake_encoder([])  # noqa: SLF001

    result = await reranker.rerank("query", [])
    assert result == []
    # The encoder was not actually invoked because the call short-circuited.
    reranker._encoder.predict.assert_not_called()  # noqa: SLF001


@pytest.mark.asyncio
async def test_rerank_passes_query_and_chunk_text_pairs() -> None:
    """The adapter must build (query, chunk.text) pairs in input order."""
    reranker = CrossEncoderReranker(RerankerConfig(provider="cross_encoder", model="m"))
    reranker._encoder = _fake_encoder([0.5, 0.5])  # noqa: SLF001

    chunks = [_make_chunk(0, text="first"), _make_chunk(1, text="second")]
    await reranker.rerank("the query", chunks)

    reranker._encoder.predict.assert_called_once()  # noqa: SLF001
    args, kwargs = reranker._encoder.predict.call_args  # noqa: SLF001
    assert args[0] == [("the query", "first"), ("the query", "second")]
    assert kwargs["batch_size"] == 32  # RerankerConfig default
    assert kwargs["show_progress_bar"] is False
