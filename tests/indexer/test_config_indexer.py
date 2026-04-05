import importlib
import sys
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
