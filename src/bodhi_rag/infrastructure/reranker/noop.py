"""
No-op reranker adapter.

Passes the input through unchanged. This is the default `provider`
in `RerankerConfig` and is what the system uses when reranking
is disabled. It exists as a real adapter (not a `None`) so the
Container can wire it like any other port and the facade can call
`reranker.rerank(...)` unconditionally.

Like every RerankerPort implementation, it exposes the configured
`overfetch_factor` so the calling pipeline can use it (e.g.,
fetch top_k * overfetch_factor candidates from the vector store
before calling `rerank`). The NoOpReranker does not itself use
the factor — the truncation to `top_k` is the only thing it does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.application.config import RerankerConfig
    from bodhi_rag.domain.entities import RetrievedDocument


class NoOpReranker:
    """
    Reranker that returns its input unchanged.

    Useful as the default `provider="noop"` adapter, and as a
    deterministic stand-in in unit tests for query pipelines that
    do not exercise reranking.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._overfetch_factor = config.overfetch_factor

    @property
    def overfetch_factor(self) -> int:
        """Return the configured overfetch factor (read-only for the pipeline)."""
        return self._overfetch_factor

    async def rerank(
        self,
        query: str,  # noqa: ARG002  # protocol signature: query not used in no-op
        chunks: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Return the input chunks in their original order.

        Args:
            query: Ignored. Present for protocol compatibility.
            chunks: Chunks to pass through.
            top_k: If set, return at most this many chunks.

        Returns:
            `chunks` (or `chunks[:top_k]` if `top_k` is set), unchanged.

        """
        if top_k is None:
            return list(chunks)
        return list(chunks[:top_k])

