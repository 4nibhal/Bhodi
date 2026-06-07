import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class ConfigIndexerCompatibilityTest(TestCase):
    def test_import_does_not_load_legacy_config(self) -> None:
        sys.modules.pop("indexer.config_indexer", None)
        sys.modules.pop("bhodi_doc_analyzer.config", None)

        importlib.import_module("indexer.config_indexer")

        self.assertNotIn("bhodi_doc_analyzer.config", sys.modules)

    def test_initialize_vectorstore_delegates_to_platform_runtime(self) -> None:
        module = importlib.import_module("indexer.config_indexer")
        vectorstore = object()
        retriever = object()

        with patch.object(
            module,
            "initialize_persistent_runtime",
            return_value=(vectorstore, retriever),
        ) as initialize_runtime:
            returned_vectorstore, returned_retriever = module.initialize_vectorstore(
                "/tmp/chroma_db",
                embeddings_factory=lambda: object(),
            )

        self.assertIs(returned_vectorstore, vectorstore)
        self.assertIs(returned_retriever, retriever)
        self.assertEqual(initialize_runtime.call_args.args[0], "/tmp/chroma_db")
        self.assertIsNotNone(initialize_runtime.call_args.kwargs["embeddings_factory"])


def test_persist_directory_honors_environment_override() -> None:
    sys.modules.pop("indexer.config_indexer", None)

    with patch.dict(
        os.environ,
        {"BHODI_INDEX_PERSIST_DIRECTORY": "/tmp/custom-index-path"},
        clear=False,
    ):
        module = importlib.import_module("indexer.config_indexer")

    assert module.PERSIST_DIRECTORY == "/tmp/custom-index-path"


def test_persist_directory_defaults_to_cwd_chroma_db() -> None:
    sys.modules.pop("indexer.config_indexer", None)

    with TemporaryDirectory() as tmpdir:
        with patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("BHODI_INDEX_PERSIST_DIRECTORY", None)
                module = importlib.import_module("indexer.config_indexer")

    assert module.PERSIST_DIRECTORY.endswith("chroma_db")
    assert module.PERSIST_DIRECTORY == str(Path(tmpdir) / "chroma_db")


def test_persistent_vectorstore_proxy_is_lazy_on_import() -> None:
    sys.modules.pop("indexer.config_indexer", None)
    vectorstore = SimpleNamespace(marker="ready")

    with patch(
        "bhodi_platform.indexing.runtime.get_persistent_vectorstore",
        return_value=vectorstore,
    ) as get_persistent_vectorstore:
        module = importlib.import_module("indexer.config_indexer")
        get_persistent_vectorstore.assert_not_called()

    assert module.persistent_vectorstore.marker == "ready"
    get_persistent_vectorstore.assert_called_once_with()
