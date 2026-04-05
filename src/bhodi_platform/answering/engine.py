from __future__ import annotations

from bhodi_platform.answering.application import AnsweringService
from bhodi_platform.application.models import AnswerQueryRequest, AnswerQueryResponse


class AnsweringEngine:
    def __init__(self, service: AnsweringService) -> None:
        self._service = service

    def answer(self, request: AnswerQueryRequest) -> AnswerQueryResponse:
        if hasattr(self._service, "answer_query"):
            return self._service.answer_query(request)

        state: dict[str, Any] = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "input": request.user_input,
            "conversation_id": request.conversation_id,
            "context": "",
            "answer": "",
        }
        result = self._service.execute(state)  # type: ignore[arg-type]
        return AnswerQueryResponse(
            answer_text=str(result.get("answer", "No response")),
            context=str(result.get("context", "")),
        )
