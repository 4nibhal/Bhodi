"""Query answering use case."""

from __future__ import annotations

from bhodi_platform.application.models import QueryRequest, QueryResponse
from bhodi_platform.ports.conversation_memory import ConversationMemoryPort
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.llm import LLMPort
from bhodi_platform.ports.vector_store import VectorStorePort


class QueryAnswerUseCase:
    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        conversation_memory: ConversationMemoryPort,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._llm = llm
        self._conversation_memory = conversation_memory

    async def execute(self, request: QueryRequest) -> QueryResponse:
        pass
