from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry
from bhodi_platform.answering.settings import AnsweringSettings
from bhodi_platform.retrieval.runtime import build_embeddings as build_shared_embeddings


_LOCAL_LLM_EXTRA_MESSAGE = (
    "Local GGUF answering requires the optional `local-llm` extra backed by "
    "`llama-cpp-python`. Install it with `uv sync --extra local-llm` or "
    "`uv sync --no-dev --extra local-llm`."
)


def _tokenizer_class() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer


def _sequence_model_class() -> Any:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification


def _pipeline_factory() -> Any:
    from transformers import pipeline

    return pipeline


def _llm_class() -> Any:
    from langchain_community.chat_models import ChatLlamaCpp

    return ChatLlamaCpp


def _vectorstore_class() -> Any:
    from langchain_chroma import Chroma

    return Chroma


@dataclass(frozen=True, slots=True)
class AnsweringRuntime:
    tokenizer: Any
    llm: Any
    embeddings: Any
    vectorstore: Any
    retriever: Any
    sequencer: Any
    reranker: Any


def build_tokenizer(settings: AnsweringSettings) -> Any:
    os.environ["TOKENIZERS_PARALLELISM"] = settings.tokenizers_parallelism
    return _tokenizer_class().from_pretrained(settings.tokenizer_model, use_fast=True)


def build_llm(settings: AnsweringSettings) -> Any:
    try:
        return _llm_class()(
            model_path=str(settings.local_model_path),
            temperature=settings.llm_temperature,
            n_ctx=settings.llm_context_window,
            n_gpu_layers=settings.llm_gpu_layers,
            n_batch=settings.llm_batch_size,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            verbose=settings.llm_verbose,
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "llama_cpp":
            raise RuntimeError(_LOCAL_LLM_EXTRA_MESSAGE) from error
        raise


def build_vectorstore(settings: AnsweringSettings, embeddings: Any) -> Any:
    return _vectorstore_class()(embedding_function=embeddings, persist_directory=None)


def build_retriever(settings: AnsweringSettings, vectorstore: Any) -> Any:
    return vectorstore.as_retriever(search_kwargs={"k": settings.volatile_retriever_k})


def build_sequencer(settings: AnsweringSettings) -> Any:
    return _pipeline_factory()(
        "summarization",
        model=settings.summarizer_model,
        tokenizer=settings.summarizer_model,
        device=-1,
    )


def build_reranker(settings: AnsweringSettings) -> Any:
    tokenizer = _tokenizer_class().from_pretrained(settings.reranker_model)
    model = _sequence_model_class().from_pretrained(settings.reranker_model)
    return _pipeline_factory()(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )


def build_embeddings(settings: AnsweringSettings) -> Any:
    return build_shared_embeddings(
        settings.embeddings_model,
        settings.embeddings_device,
    )


def initialize_runtime(settings: AnsweringSettings | None = None) -> AnsweringRuntime:
    resolved_settings = settings or AnsweringSettings.from_environment()
    tokenizer = build_tokenizer(resolved_settings)
    llm = build_llm(resolved_settings)
    embeddings = build_embeddings(resolved_settings)
    vectorstore = build_vectorstore(resolved_settings, embeddings)
    retriever = build_retriever(resolved_settings, vectorstore)
    sequencer = build_sequencer(resolved_settings)
    reranker = build_reranker(resolved_settings)
    return AnsweringRuntime(
        tokenizer=tokenizer,
        llm=llm,
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        sequencer=sequencer,
        reranker=reranker,
    )


_runtime_registry = RuntimeRegistry(initialize_runtime)


def start_runtime() -> AnsweringRuntime:
    return _runtime_registry.start()


def get_runtime() -> AnsweringRuntime:
    return _runtime_registry.get()


def reset_runtime() -> None:
    _runtime_registry.reset()


def stop_runtime() -> None:
    _runtime_registry.stop()


def get_tokenizer() -> Any:
    return get_runtime().tokenizer


def get_llm() -> Any:
    return get_runtime().llm


def get_embeddings() -> Any:
    return get_runtime().embeddings


def get_vectorstore() -> Any:
    return get_runtime().vectorstore


def get_retriever() -> Any:
    return get_runtime().retriever


def get_sequencer() -> Any:
    return get_runtime().sequencer


def get_reranker() -> Any:
    return get_runtime().reranker


class LazyObjectProxy:
    def __init__(self, factory: Any) -> None:
        self._factory = factory

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory()(*args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._factory(), attribute)


tokenizer = LazyObjectProxy(get_tokenizer)
llm = LazyObjectProxy(get_llm)
embeddings = LazyObjectProxy(get_embeddings)
vectorstore = LazyObjectProxy(get_vectorstore)
retriever = LazyObjectProxy(get_retriever)
sequencer = LazyObjectProxy(get_sequencer)
reranker = LazyObjectProxy(get_reranker)
