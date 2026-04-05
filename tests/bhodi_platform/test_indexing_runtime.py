import importlib
import sys
from unittest import TestCase
from unittest.mock import patch


class PersistentIndexingRuntimeTest(TestCase):
    def test_initialize_runtime_uses_platform_embeddings_without_legacy_imports(
        self,
    ) -> None:
        sys.modules.pop("bhodi_doc_analyzer.config", None)
        sys.modules.pop("bhodi_platform.answering.runtime", None)
        module = importlib.import_module("bhodi_platform.indexing.runtime")
        embeddings = object()
        vectorstore = object()
        retriever = object()

        with patch.object(module, "get_embeddings", return_value=embeddings):
            with patch.object(
                module, "build_vectorstore", return_value=vectorstore
            ) as build_vectorstore:
                with patch.object(
                    module, "build_retriever", return_value=retriever
                ) as build_retriever:
                    returned_vectorstore, returned_retriever = (
                        module.initialize_persistent_runtime("/tmp/chroma_db")
                    )

        self.assertIs(returned_vectorstore, vectorstore)
        self.assertIs(returned_retriever, retriever)
        self.assertEqual(
            str(build_vectorstore.call_args.args[0].persist_directory), "/tmp/chroma_db"
        )
        self.assertIs(build_vectorstore.call_args.args[1], embeddings)
        build_retriever.assert_called_once_with(
            vectorstore, build_vectorstore.call_args.args[0]
        )
        self.assertNotIn("bhodi_doc_analyzer.config", sys.modules)
        self.assertNotIn("bhodi_platform.answering.runtime", sys.modules)

    def test_runtime_uses_shared_retrieval_embeddings_seam(self) -> None:
        module = importlib.import_module("bhodi_platform.indexing.runtime")

        self.assertEqual(
            module.get_embeddings.__module__, "bhodi_platform.retrieval.runtime"
        )
