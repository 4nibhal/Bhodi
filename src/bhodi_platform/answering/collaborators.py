from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from bhodi_platform.answering.ports import (
    LanguageModelPort,
    LogWriter,
    RerankerPort,
    RetrieverPort,
    SummarizerPort,
    TokenizerPort,
)
from bhodi_platform.answering.settings import AnsweringSettings
from bhodi_platform.application.citation_models import DocumentToContext
from bhodi_platform.application.models import (
    RetrievedDocumentDiagnostics,
    RetrievalDiagnostics,
    RetrievalTruncationDiagnostics,
)


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    input: str
    conversation_id: str | None
    context: str
    answer: str


class ContextUpdate(TypedDict, total=False):
    context: str
    retrieval: Any


class AnswerUpdate(TypedDict):
    answer: str


class RetrievalCollaborator(Protocol):
    def retrieve_context(self, state: AgentState) -> ContextUpdate:
        """Build the context window for a user query."""
        ...

    def summarize_text(self, text: str) -> str:
        """Summarize long text while preserving document identity."""
        ...

    def rerank_documents(self, query: str, docs: list[Any]) -> list[Any]:
        """Return documents ordered by query relevance."""
        ...

    def sequence_documents(
        self, query: str, docs: list[Any]
    ) -> list[DocumentToContext]:
        """Prepare retrieved documents for prompt assembly."""
        ...


class GenerationCollaborator(Protocol):
    def fast_tokenize(self, texts: Any, max_length: int | None = None) -> Any:
        """Tokenize prompt content with the configured model tokenizer."""
        ...

    def generate_response(self, state: AgentState) -> AnswerUpdate:
        """Generate an answer for the provided state."""
        ...

    def parse_answer_response(
        self,
        raw_response: Any,
        *,
        error_prefix: str,
        coerce_plain_text: bool,
    ) -> Any:
        """Parse model output while preserving compatibility fallbacks."""
        ...

    def refine_prompt(self, context: str, user_input: str) -> str:
        """Build the final prompt text sent to the model."""
        ...

    def transform_message(self, msg: dict[str, Any]) -> Any:
        """Map legacy message roles into model-native messages."""
        ...


class AssistantAnswer(BaseModel):
    answer: str = Field(..., description="The assistant's answer content")


answer_parser = PydanticOutputParser(pydantic_object=AssistantAnswer)


class DefaultRetrievalCollaborator:
    def __init__(
        self,
        *,
        settings: AnsweringSettings,
        session_retriever: RetrieverPort,
        corpus_retriever: RetrieverPort,
        conversation_retriever_factory: Callable[[str | None], RetrieverPort],
        tokenizer: TokenizerPort,
        sequencer: SummarizerPort,
        reranker: RerankerPort,
    ) -> None:
        self._settings = settings
        self._session_retriever = session_retriever
        self._corpus_retriever = corpus_retriever
        self._conversation_retriever_factory = conversation_retriever_factory
        self._tokenizer = tokenizer
        self._sequencer = sequencer
        self._reranker = reranker

    def retrieve_context(self, state: AgentState) -> ContextUpdate:
        query = state["input"]
        conversation_id = state.get("conversation_id")
        retrieved_docs = [
            ("session", doc) for doc in self._session_retriever.invoke(query)
        ]
        if conversation_id is not None:
            conversation_retriever = self._conversation_retriever_factory(
                conversation_id
            )
            retrieved_docs.extend(
                ("conversation", doc) for doc in conversation_retriever.invoke(query)
            )
        retrieved_docs.extend(
            ("corpus", doc) for doc in self._corpus_retriever.invoke(query)
        )
        all_docs = [document for _, document in retrieved_docs]
        sequenced_docs = self.sequence_documents(query, all_docs)

        context, docs_in_context = self._assemble_context_with_token_budget(
            sequenced_docs
        )

        return {
            "context": context,
            "retrieval": self._build_retrieval_diagnostics(
                retrieved_docs=retrieved_docs,
                sequenced_docs=sequenced_docs,
                docs_in_context=docs_in_context,
                context=context,
            ),
        }

    def _assemble_context_with_token_budget(
        self, docs: list[DocumentToContext]
    ) -> tuple[str, list[DocumentToContext]]:
        """Assemble context from documents using token budget.

        Returns:
            Tuple of (assembled_context, list of docs actually used in context)
        """
        available_tokens = self._settings.context_token_limit
        context_parts: list[str] = []
        docs_in_context: list[DocumentToContext] = []

        for i, doc in enumerate(docs):
            doc_tokens = self._count_tokens(doc.page_content)
            separator = "\n" if i > 0 else ""
            separator_tokens = self._count_tokens(separator) if i > 0 else 0

            if doc_tokens + separator_tokens <= available_tokens:
                context_parts.append(separator + doc.page_content)
                docs_in_context.append(doc)
                available_tokens -= doc_tokens + separator_tokens
            else:
                break

        return "".join(context_parts), docs_in_context

    def summarize_text(self, text: str) -> str:
        if len(text) < self._settings.raw_summary_char_limit:
            return text
        summary = self._sequencer(
            text,
            max_length=self._settings.summarizer_max_length,
            min_length=self._settings.summarizer_min_length,
            truncation=True,
        )
        return summary[0]["summary_text"]

    def rerank_documents(self, query: str, docs: list[Any]) -> list[Any]:
        scored_docs = []
        for doc in docs:
            page_content = self._validated_page_content(doc)
            result = self._reranker(
                (query, page_content),
                padding=True,
                truncation=True,
                max_length=self._settings.reranker_max_length,
            )
            score = result[0]["score"] if isinstance(result, list) else result["score"]
            scored_docs.append((score, doc))
        return [
            doc
            for _, doc in sorted(scored_docs, key=lambda item: item[0], reverse=True)
        ]

    def sequence_documents(
        self, query: str, docs: list[Any]
    ) -> list[DocumentToContext]:
        """Prepare documents for context assembly without mutating source docs.

        Returns a list of DocumentToContext with summaries applied where needed.
        """
        reranked_docs = self.rerank_documents(query, docs)
        result: list[DocumentToContext] = []

        for doc in reranked_docs:
            page_content = self._validated_page_content(doc)
            if (
                self._count_tokens(page_content)
                > self._settings.document_summary_token_limit
            ):
                summary = self.summarize_text(page_content)
                doc_to_context = DocumentToContext.from_retrieved_document(doc)
                result.append(doc_to_context.with_summary(summary))
            else:
                result.append(DocumentToContext.from_retrieved_document(doc))

        return result

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    @staticmethod
    def _validated_page_content(document: Any) -> str:
        page_content = getattr(document, "page_content", None)
        if not isinstance(page_content, str):
            raise TypeError(
                "Retrieved documents must expose string page_content to preserve "
                "content and metadata for future citation support."
            )
        return page_content

    def _build_retrieval_diagnostics(
        self,
        *,
        retrieved_docs: list[tuple[str, Any]],
        sequenced_docs: list[DocumentToContext],
        docs_in_context: list[DocumentToContext],
        context: str,
    ) -> RetrievalDiagnostics:
        # Build origin map from original document IDs
        origins = {id(doc): origin for origin, doc in retrieved_docs}
        # Track which source_ids are in context
        docs_in_context_ids = {doc.source_id for doc in docs_in_context}

        diagnostics: list[RetrievedDocumentDiagnostics] = []
        context_tokens = self._count_tokens(context)
        total_original_tokens = sum(
            self._count_tokens(doc.original_content) for doc in sequenced_docs
        )

        for rank, doc_to_context in enumerate(sequenced_docs, start=1):
            doc_tokens = self._count_tokens(doc_to_context.page_content)
            original_tokens = self._count_tokens(doc_to_context.original_content)
            used_in_context = doc_to_context.source_id in docs_in_context_ids
            source_id = doc_to_context.source_id

            diagnostics.append(
                RetrievedDocumentDiagnostics(
                    rank=rank,
                    retriever_origin=(
                        origins.get(source_id, "unknown")
                        if source_id is not None
                        else "unknown"
                    ),
                    source=self._extract_source(doc_to_context.metadata),
                    metadata=doc_to_context.metadata,
                    summarized=doc_to_context.is_summarized,
                    used_in_context=used_in_context,
                    preview=doc_to_context.page_content[:200],
                    truncation=RetrievalTruncationDiagnostics(
                        original_length=original_tokens,
                        returned_length=doc_tokens if used_in_context else 0,
                        truncated=doc_tokens < original_tokens,
                    ),
                )
            )

        return RetrievalDiagnostics(
            documents=tuple(diagnostics),
            context=RetrievalTruncationDiagnostics(
                original_length=total_original_tokens,
                returned_length=context_tokens,
                truncated=total_original_tokens > context_tokens,
            ),
        )

    @staticmethod
    def _normalized_metadata(document: Any) -> dict[str, Any]:
        metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[str(key)] = value
            else:
                normalized[str(key)] = str(value)
        return normalized

    @staticmethod
    def _extract_source(metadata: dict[str, Any]) -> str | None:
        source = metadata.get("source")
        return source if isinstance(source, str) else None


class DefaultGenerationCollaborator:
    def __init__(
        self,
        *,
        settings: AnsweringSettings,
        llm: LanguageModelPort,
        tokenizer: TokenizerPort,
        summarizer: SummarizerPort,
        log_writer: LogWriter,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._tokenizer = tokenizer
        self._summarizer = summarizer
        self._log_writer = log_writer

    def generate_response(self, state: AgentState) -> AnswerUpdate:
        prompt_text = self.refine_prompt(state.get("context", ""), state["input"])
        self.fast_tokenize(prompt_text)
        transformed_messages = [
            self.transform_message(msg) for msg in state["messages"]
        ]
        messages = transformed_messages + [HumanMessage(content=prompt_text)]
        raw_response = self._llm.invoke(messages)
        self._log_writer(f"Raw response: {raw_response}")
        answer_text = self.parse_answer_response(
            raw_response,
            error_prefix="Parsing error",
            coerce_plain_text=True,
        )
        return {"answer": answer_text}

    def parse_answer_response(
        self,
        raw_response: Any,
        *,
        error_prefix: str,
        coerce_plain_text: bool,
    ) -> Any:
        try:
            structured_obj = answer_parser.parse(raw_response)
            answer_text: Any = structured_obj.answer
        except ValueError as error:
            self._log_writer(f"{error_prefix}: {error}")
            answer_text = raw_response

        if not coerce_plain_text:
            return answer_text

        if not isinstance(answer_text, str):
            try:
                answer_text = answer_text.content
            except AttributeError:
                answer_text = str(answer_text)
        return answer_text

    def refine_prompt(self, context: str, user_input: str) -> str:
        if self._count_tokens(context) > self._settings.prompt_summary_token_limit:
            context = self._summarize_text(context)
        prompt = f"Context: {context}\nQuestion: {user_input}"
        print(f"Prompt tokens: {self._count_tokens(prompt)}")
        return prompt

    @staticmethod
    def transform_message(msg: dict[str, Any]) -> Any:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ["question", "human", "user"]:
            return HumanMessage(content=content)
        if role in ["answer", "assistant", "ai"]:
            return AIMessage(content=content)
        return HumanMessage(content=content)

    def fast_tokenize(self, texts: Any, max_length: int | None = None) -> Any:
        return self._tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_length or self._settings.tokenizer_max_length,
            return_tensors="pt",
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _summarize_text(self, text: str) -> str:
        if len(text) < self._settings.raw_summary_char_limit:
            return text
        summary = self._summarizer(
            text,
            max_length=self._settings.summarizer_max_length,
            min_length=self._settings.summarizer_min_length,
            truncation=True,
        )
        return summary[0]["summary_text"]
