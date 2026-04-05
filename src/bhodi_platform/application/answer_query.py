from __future__ import annotations

from bhodi_platform.application.models import AnswerQueryRequest, AnswerQueryResponse
from bhodi_platform.ports.answering import (
    AnswerQueryEnginePort,
    ConversationMemoryPort,
)


class AnswerQueryUseCase:
    def __init__(
        self,
        engine: AnswerQueryEnginePort,
        conversation_memory: ConversationMemoryPort | None = None,
    ) -> None:
        self._engine = engine
        self._conversation_memory = conversation_memory

    def execute(self, request: AnswerQueryRequest) -> AnswerQueryResponse:
        response = self._engine.answer(request)
        if self._conversation_memory is not None:
            self._conversation_memory.append_turn(
                request.user_input,
                response.answer_text,
                conversation_id=request.conversation_id,
            )
        return response
