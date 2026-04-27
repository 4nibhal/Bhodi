"""
OpenAI LLM adapter.

Generates text using OpenAI's chat completions API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from bhodi_platform.domain.exceptions import LLMError
from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import LLMConfig
    from bhodi_platform.ports.vector_store import RetrievedDocument


class OpenAILLMAdapter:
    """
    OpenAI adapter for LLM text generation.

    Uses OpenAI's chat.completions API to generate responses.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None
        self._model = config.model or self.DEFAULT_MODEL

    async def _ensure_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY") or self._config.extra.get("api_key")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self._client = AsyncOpenAI(api_key=api_key)

    @traced("openai.llm.generate")
    async def generate(
        self,
        prompt: str,
        **kwargs: str | int | float,
    ) -> str:
        """Generate text from a prompt using OpenAI."""
        await self._ensure_client()

        temperature = kwargs.get("temperature", self._config.temperature)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens or 2048)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError("generate", str(exc)) from exc

        return response.choices[0].message.content or ""

    @traced("openai.llm.generate_with_context")
    async def generate_with_context(
        self,
        query: str,
        contexts: list[RetrievedDocument],
        **kwargs: str | int | float,
    ) -> str:
        """Generate answer given a query and retrieved context."""
        context_parts = []
        for i, doc in enumerate(contexts, 1):
            context_parts.append(f"[Document {i}]\n{doc.text}")

        context_str = "\n\n".join(context_parts)

        prompt = f"""Answer the question based on the provided context.

Context:
{context_str}

Question: {query}

Answer:"""

        return await self.generate(prompt, **kwargs)
