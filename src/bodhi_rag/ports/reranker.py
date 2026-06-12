"""
Reranker port definition.

Defines the contract for reranking adapters that reorder vector-store
retrieval candidates by query relevance.

The no-hardcoded-defaults policy established in `RerankerConfig`
(Wave 1, config.py) is the only place a reranker's model name lives.
Adapters read it from config; they never invent one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bodhi_rag.ports.vector_store import RetrievedDocument


@runtime_checkable
class RerankerPort(Protocol):
    """
    Protocol for reranking strategies.

    Adapters implementing this port take the candidate set returned
    by a vector store and reorder it by query relevance, optionally
    trimming to `top_k`.

    The score on each returned `RetrievedDocument` is updated to
    reflect the reranker's relevance signal (higher = more relevant).
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Rerank chunks by query relevance.

        Args:
            query: The original user query.
            chunks: Candidate chunks from the vector store, in retrieval order.
            top_k: If set, return at most this many chunks after reranking.
                If None, return all chunks (in reranked order).

        Returns:
            Chunks in reranked order (most relevant first). The `score`
            field on each chunk is the reranker's relevance score, not
            the original vector-store similarity.

        Raises:
            RerankerError: If reranking fails.

        """
        ...
