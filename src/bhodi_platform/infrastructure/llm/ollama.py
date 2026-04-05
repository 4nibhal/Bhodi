"""
Ollama LLM adapter.

Generates text using Ollama local LLM API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import LLMConfig
    from bhodi_platform.ports.vector_store import RetrievedDocument


class OllamaLLMAdapter:
    """
    Ollama adapter for local LLM generation.

    Uses Ollama's REST API to generate text.
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None
        self._model = config.model or self.DEFAULT_MODEL
        self._base_url = config.extra.get("base_url", self.DEFAULT_BASE_URL)

    async def _ensure_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=60.0,
            )

    @traced("ollama.generate")
    async def generate(
        self,
        prompt: str,
        **kwargs: str | int | float,
    ) -> str:
        """Generate text from a prompt using Ollama."""
        await self._ensure_client()

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        response = await self._client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return data.get("response", "")

    @traced("ollama.generate_with_context")
    async def generate_with_context(
        self,
        query: str,
        contexts: list[RetrievedDocument],
        **kwargs: str | int | float,
    ) -> str:
        """Generate answer given a query and retrieved context."""
        # Build context string from retrieved documents
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
