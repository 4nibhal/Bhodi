"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from bhodi_platform.answering.runtime import (
    embeddings,
    get_embeddings,
    get_llm,
    get_reranker,
    get_retriever,
    get_runtime,
    get_sequencer,
    get_tokenizer,
    get_vectorstore,
    llm,
    reranker,
    retriever,
    sequencer,
    tokenizer,
    vectorstore,
)
from bhodi_platform.answering.settings import AnsweringSettings

LOCAL_MODEL = str(AnsweringSettings.from_environment().local_model_path)

__all__ = [
    "LOCAL_MODEL",
    "embeddings",
    "get_embeddings",
    "get_llm",
    "get_reranker",
    "get_retriever",
    "get_runtime",
    "get_sequencer",
    "get_tokenizer",
    "get_vectorstore",
    "llm",
    "reranker",
    "retriever",
    "sequencer",
    "tokenizer",
    "vectorstore",
]


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
