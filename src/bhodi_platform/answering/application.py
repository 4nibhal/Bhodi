from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from bhodi_platform.answering.collaborators import (
    AgentState,
    AssistantAnswer,
    AnswerUpdate,
    ContextUpdate,
    DefaultGenerationCollaborator,
    DefaultRetrievalCollaborator,
    GenerationCollaborator,
    RetrievalCollaborator,
    answer_parser,
)
from bhodi_platform.answering.memory import store_conversation_turn
from bhodi_platform.answering.ports import (
    LanguageModelPort,
    LogWriter,
    RerankerPort,
    RetrieverPort,
    SummarizerPort,
    TokenizerPort,
    VectorStorePort,
)
from bhodi_platform.application.models import (
    AnswerQueryRequest,
    AnswerQueryResponse,
    ConversationMessage,
)
from bhodi_platform.answering.settings import AnsweringSettings


class AnsweringService:
    def __init__(
        self,
        *,
        settings: AnsweringSettings,
        volatile_retriever: RetrieverPort,
        persistent_retriever: RetrieverPort,
        conversation_retriever_factory: Callable[[str | None], RetrieverPort],
        llm: LanguageModelPort,
        tokenizer: TokenizerPort,
        sequencer: SummarizerPort,
        reranker: RerankerPort,
        log_writer: LogWriter,
        retrieval: RetrievalCollaborator | None = None,
        generation: GenerationCollaborator | None = None,
    ) -> None:
        self._settings = settings
        self._volatile_retriever = volatile_retriever
        self._persistent_retriever = persistent_retriever
        self._llm = llm
        self._tokenizer = tokenizer
        self._sequencer = sequencer
        self._reranker = reranker
        self._log_writer = log_writer
        self._retrieval = retrieval or DefaultRetrievalCollaborator(
            settings=settings,
            session_retriever=volatile_retriever,
            corpus_retriever=persistent_retriever,
            conversation_retriever_factory=conversation_retriever_factory,
            tokenizer=tokenizer,
            sequencer=sequencer,
            reranker=reranker,
        )
        self._generation = generation or DefaultGenerationCollaborator(
            settings=settings,
            llm=llm,
            tokenizer=tokenizer,
            summarizer=sequencer,
            log_writer=log_writer,
        )

    def fast_tokenize(self, texts: Any, max_length: int | None = None) -> Any:
        return self._generation.fast_tokenize(texts, max_length)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def retrieve_context(self, state: AgentState) -> ContextUpdate:
        return self._retrieval.retrieve_context(state)

    def generate_response(self, state: AgentState) -> AnswerUpdate:
        self._sync_generation_log_writer()
        return self._generation.generate_response(state)

    def parse_answer_response(
        self,
        raw_response: Any,
        *,
        error_prefix: str,
        coerce_plain_text: bool,
    ) -> Any:
        self._sync_generation_log_writer()
        return self._generation.parse_answer_response(
            raw_response,
            error_prefix=error_prefix,
            coerce_plain_text=coerce_plain_text,
        )

    def summarize_text(self, text: str) -> str:
        return self._retrieval.summarize_text(text)

    def refine_prompt(self, context: str, user_input: str) -> str:
        return self._generation.refine_prompt(context, user_input)

    def transform_message(self, msg: dict[str, Any]) -> Any:
        return self._generation.transform_message(msg)

    def rerank_documents(self, query: str, docs: list[Any]) -> list[Any]:
        return self._retrieval.rerank_documents(query, docs)

    def sequence_documents(self, query: str, docs: list[Any]) -> list[Any]:
        return self._retrieval.sequence_documents(query, docs)

    def execute(self, state: AgentState) -> AgentState:
        state["context"] = self.retrieve_context(state).get("context", "")
        state["answer"] = self.generate_response(state)["answer"]
        return state

    def answer_query(self, request: AnswerQueryRequest) -> AnswerQueryResponse:
        state = self._build_state(request)
        context_update = self.retrieve_context(state)
        state["context"] = context_update.get("context", "")
        answer_update = self.generate_response(state)
        state["answer"] = answer_update["answer"]
        return AnswerQueryResponse(
            answer_text=str(state.get("answer", "No response")),
            context=str(state.get("context", "")),
            retrieval=context_update.get("retrieval"),
        )

    @staticmethod
    def _build_state(request: AnswerQueryRequest) -> AgentState:
        state: AgentState = cast(
            AgentState,
            {
                "messages": [
                    AnsweringService._serialize_message(message)
                    for message in request.messages
                ],
                "input": request.user_input,
                "conversation_id": request.conversation_id,
                "context": "",
                "answer": "",
            },
        )
        return state

    @staticmethod
    def _serialize_message(message: ConversationMessage) -> dict[str, str]:
        return {"role": message.role, "content": message.content}

    def _sync_generation_log_writer(self) -> None:
        if hasattr(self._generation, "_log_writer"):
            self._generation._log_writer = self._log_writer  # type: ignore[attr-defined]


def build_default_answering_service(log_writer: LogWriter) -> AnsweringService:
    from bhodi_platform.answering.runtime import (
        get_llm,
        get_reranker,
        get_retriever,
        get_sequencer,
        get_tokenizer,
    )
    from bhodi_platform.conversation.runtime import (
        get_persistent_retriever as get_persistent_conversation_retriever,
    )
    from bhodi_platform.indexing.runtime import get_persistent_retriever

    settings = AnsweringSettings.from_environment()
    return AnsweringService(
        settings=settings,
        volatile_retriever=get_retriever(),
        persistent_retriever=get_persistent_retriever(),
        conversation_retriever_factory=get_persistent_conversation_retriever,
        llm=get_llm(),
        tokenizer=get_tokenizer(),
        sequencer=get_sequencer(),
        reranker=get_reranker(),
        log_writer=log_writer,
    )
