# ruff: noqa: INP001

import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest


class ConfigIndexerCompatibilityTest(TestCase):
    def test_import_does_not_load_legacy_config(self) -> None:
        sys.modules.pop("indexer.config_indexer", None)
        sys.modules.pop("bhodi_doc_analyzer.config", None)

        importlib.import_module("indexer.config_indexer")

        if "bhodi_doc_analyzer.config" in sys.modules:
            pytest.fail(
                "Importing indexer.config_indexer should not import bhodi_doc_analyzer.config",
            )

    def test_initialize_vectorstore_delegates_to_platform_runtime(self) -> None:
        module = importlib.import_module("indexer.config_indexer")
        persist_directory = "chroma_db"
        vectorstore = object()
        retriever = object()

        with patch.object(
            module,
            "initialize_persistent_runtime",
            return_value=(vectorstore, retriever),
        ) as initialize_runtime:
            returned_vectorstore, returned_retriever = module.initialize_vectorstore(
                persist_directory,
                embeddings_factory=object,
            )

        if returned_vectorstore is not vectorstore:
            pytest.fail("initialize_vectorstore should return the delegated vectorstore")

        if returned_retriever is not retriever:
            pytest.fail("initialize_vectorstore should return the delegated retriever")

        initialize_runtime.assert_called_once_with(
            persist_directory,
            embeddings_factory=object,
        )


def test_persist_directory_honors_environment_override() -> None:
    sys.modules.pop("indexer.config_indexer", None)
    custom_index_path = "custom-index-path"

    with patch.dict(
        os.environ,
        {"BHODI_INDEX_PERSIST_DIRECTORY": custom_index_path},
        clear=False,
    ):
        module = importlib.import_module("indexer.config_indexer")

    if custom_index_path != module.PERSIST_DIRECTORY:
        pytest.fail(
            f"Expected PERSIST_DIRECTORY={custom_index_path!r}, got {module.PERSIST_DIRECTORY!r}",
        )


def test_persist_directory_defaults_to_cwd_chroma_db() -> None:
    sys.modules.pop("indexer.config_indexer", None)

    with (
        TemporaryDirectory() as tmpdir,
        patch("pathlib.Path.cwd", return_value=Path(tmpdir)),
        patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("BHODI_INDEX_PERSIST_DIRECTORY", None)
        module = importlib.import_module("indexer.config_indexer")

    expected_directory = str(Path(tmpdir) / "chroma_db")

    if not module.PERSIST_DIRECTORY.endswith("chroma_db"):
        pytest.fail(
            f"Expected PERSIST_DIRECTORY to end with 'chroma_db', got {module.PERSIST_DIRECTORY!r}",
        )

    if expected_directory != module.PERSIST_DIRECTORY:
        pytest.fail(
            f"Expected PERSIST_DIRECTORY={expected_directory!r}, got {module.PERSIST_DIRECTORY!r}",
        )


def test_persistent_vectorstore_proxy_is_lazy_on_import() -> None:
    sys.modules.pop("indexer.config_indexer", None)
    vectorstore = SimpleNamespace(marker="ready")

    with patch(
        "bhodi_platform.indexing.runtime.get_persistent_vectorstore",
        return_value=vectorstore,
    ) as get_persistent_vectorstore:
        module = importlib.import_module("indexer.config_indexer")
        get_persistent_vectorstore.assert_not_called()

    if module.persistent_vectorstore.marker != "ready":
        pytest.fail("persistent_vectorstore should resolve lazily to the delegated vectorstore")

    get_persistent_vectorstore.assert_called_once_with()
