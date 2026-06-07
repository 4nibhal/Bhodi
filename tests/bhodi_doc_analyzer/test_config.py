# ruff: noqa: INP001

import importlib
import sys

import pytest


def _clear_module(name: str) -> None:
    sys.modules.pop(name, None)


def test_config_import_succeeds() -> None:
    _clear_module("bhodi_doc_analyzer.config")

    module = importlib.import_module("bhodi_doc_analyzer.config")

    assert module.__name__ == "bhodi_doc_analyzer.config"


def test_config_exposes_local_model_as_string() -> None:
    module = importlib.import_module("bhodi_doc_analyzer.config")

    assert isinstance(module.LOCAL_MODEL, str)


def test_config_public_exports_present() -> None:
    module = importlib.import_module("bhodi_doc_analyzer.config")
    expected_exports = {
        "LOCAL_MODEL",
        "get_embeddings",
        "get_llm",
        "get_reranker",
        "get_retriever",
        "get_runtime",
        "get_sequencer",
        "get_tokenizer",
        "get_vectorstore",
        "embeddings",
        "llm",
        "reranker",
        "retriever",
        "sequencer",
        "tokenizer",
        "vectorstore",
    }

    assert expected_exports <= set(module.__all__)
    for name in expected_exports:
        assert hasattr(module, name)


def test_config_unknown_attribute_raises_attribute_error() -> None:
    module = importlib.import_module("bhodi_doc_analyzer.config")

    with pytest.raises(AttributeError):
        getattr(module, "does_not_exist")
