import importlib
import sys
from types import ModuleType
from unittest.mock import Mock, patch


def _import_document_indexing_module():
    sys.modules.pop("indexer.document_indexing", None)
    sys.modules.pop("bhodi_platform.indexing.application", None)

    fake_application = ModuleType("bhodi_platform.indexing.application")

    class StubDocumentIndexingService:
        pass

    fake_application.DocumentIndexingService = StubDocumentIndexingService

    with patch.dict(
        sys.modules,
        {"bhodi_platform.indexing.application": fake_application},
    ):
        return importlib.import_module("indexer.document_indexing")


def test_load_and_index_documents_from_directory_delegates_to_service() -> None:
    document_indexing = _import_document_indexing_module()
    service = Mock()
    service.index_directory.return_value = 7
    settings = object()
    vectorstore = object()

    with patch.object(
        document_indexing,
        "_build_indexing_service",
        return_value=service,
    ) as build_indexing_service:
        with patch.object(
            document_indexing.IndexingSettings,
            "from_environment",
            return_value=settings,
        ) as from_environment:
            indexed_count = document_indexing.load_and_index_documents_from_directory(
                "/tmp/docs",
                vectorstore,
            )

    assert indexed_count == 7
    build_indexing_service.assert_called_once_with()
    from_environment.assert_called_once_with()
    service.index_directory.assert_called_once_with("/tmp/docs", vectorstore, settings)


def test_load_and_index_single_file_delegates_to_service() -> None:
    document_indexing = _import_document_indexing_module()
    service = Mock()
    service.index_file.return_value = 3
    settings = object()
    vectorstore = object()

    with patch.object(
        document_indexing,
        "_build_indexing_service",
        return_value=service,
    ) as build_indexing_service:
        with patch.object(
            document_indexing.IndexingSettings,
            "from_environment",
            return_value=settings,
        ) as from_environment:
            indexed_count = document_indexing.load_and_index_single_file(
                "/tmp/file.pdf",
                vectorstore,
            )

    assert indexed_count == 3
    build_indexing_service.assert_called_once_with()
    from_environment.assert_called_once_with()
    service.index_file.assert_called_once_with("/tmp/file.pdf", vectorstore, settings)
