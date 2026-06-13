"""
SynthesizeAnswerUseCase.

Takes the question and the retrieved context, returns the
synthesized answer text. The use case is the application-layer
abstraction the facade depends on; it currently delegates to
`LLMPort.generate_with_context` but the bounded context shape
exists so future answer-side concerns (citation extraction,
grounded-template enforcement, streaming, guardrails) can land
here without rippling into the retrieval pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhi_rag.domain.entities import RetrievedDocument
    from bodhi_rag.ports.llm import LLMPort


class SynthesizeAnswerUseCase:
    """
    Application-layer entry point for synthesizing the final answer.

    The use case shape is the right place to add cross-cutting
    concerns (telemetry spans, structured logging, answer length
    caps, refusal templates) without leaking them into adapters.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def execute(
        self,
        question: str,
        contexts: list[RetrievedDocument],
        *,
        temperature: float,
    ) -> str:
        """
        Synthesize the natural-language answer for `question` grounded in `contexts`.

        Args:
            question: The user query.
            contexts: Chunks already reranked and trimmed to top_k.
            temperature: Sampling temperature passed through to the LLM.

        Returns:
            The synthesized answer text.

        """
        return await self._llm.generate_with_context(
            question,
            contexts,
            temperature=temperature,
        )
