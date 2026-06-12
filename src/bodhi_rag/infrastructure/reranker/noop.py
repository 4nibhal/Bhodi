"""
No-op reranker adapter.

Passes the input through unchanged. This is the default `provider`
in `RerankerConfig` and is what the system uses when reranking
is disabled. It exists as a real adapter (not a `None`) so the
Container can wire it like any other port and the facade can call
`reranker.rerank(...)` unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import RetrievedDocument


class NoOpReranker:
    """
    Reranker that returns its input unchanged.

    Useful as the default `provider="noop"` adapter, and as a
    deterministic stand-in in unit tests for query pipelines that
    do not exercise reranking.
    """

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
