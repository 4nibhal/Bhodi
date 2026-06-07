from pathlib import Path
from unittest.mock import Mock

import pytest

from bhodi_platform.application import IndexDocumentsRequest, IndexDocumentsResponse
from bhodi_platform.indexing import (
    DocumentIndexingService,
    IndexingEngine,
    IndexingSettings,
    InvalidDocumentPathError,
)


def test_document_indexing_service_indexes_directories(tmp_path: Path) -> None:
    directory_path = tmp_path / "docs"
    directory_path.mkdir()
    vectorstore = object()
    settings = IndexingSettings(persist_directory=tmp_path / "chroma_db")
    service = DocumentIndexingService()
    expected_response = 7

    service.index_directory = Mock(return_value=expected_response)

    response = service.index_request(
        IndexDocumentsRequest(document_path=directory_path),
        vectorstore=vectorstore,
        settings=settings,
    )

    assert response.indexed_fragments == expected_response
    assert response.source_kind == "directory"
    assert response.resolved_path == directory_path.resolve()
    service.index_directory.assert_called_once_with(
        str(directory_path.resolve()),
        vectorstore,
        settings,
    )


def test_document_indexing_service_indexes_files(tmp_path: Path) -> None:
    file_path = tmp_path / "source.txt"
    file_path.write_text("hello", encoding="utf-8")
    vectorstore = object()
    settings = IndexingSettings(persist_directory=tmp_path / "chroma_db")
    service = DocumentIndexingService()
    expected_response = 3

    service.index_file = Mock(return_value=expected_response)

    response = service.index_request(
        IndexDocumentsRequest(document_path=file_path),
        vectorstore=vectorstore,
        settings=settings,
    )

    assert response.indexed_fragments == expected_response
    assert response.source_kind == "file"
    assert response.resolved_path == file_path.resolve()
    service.index_file.assert_called_once_with(
        str(file_path.resolve()),
        vectorstore,
        settings,
    )


def test_document_indexing_service_rejects_unknown_paths(tmp_path: Path) -> None:
    service = DocumentIndexingService()
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(InvalidDocumentPathError, match=str(missing_path.resolve())):
        service.index_request(
            IndexDocumentsRequest(document_path=missing_path),
            vectorstore=object(),
            settings=IndexingSettings(persist_directory=tmp_path / "chroma_db"),
        )


def test_indexing_engine_delegates_to_index_request(tmp_path: Path) -> None:
    file_path = tmp_path / "source.md"
    file_path.write_text("# hello", encoding="utf-8")
    request = IndexDocumentsRequest(document_path=file_path, cwd=tmp_path)
    settings = IndexingSettings(persist_directory=tmp_path / "persistent-index")
    vectorstore = object()
    retriever = object()
    service = Mock()
    expected_response = IndexDocumentsResponse(
        indexed_fragments=5,
        source_kind="file",
        resolved_path=file_path.resolve(),
    )
    service.index_request.return_value = expected_response
    runtime_factory = Mock(return_value=(vectorstore, retriever))
    settings_factory = Mock(return_value=settings)
    engine = IndexingEngine(
        service=service,
        runtime_factory=runtime_factory,
        settings_factory=settings_factory,
    )

    response = engine.index(request)

    assert response is expected_response
    settings_factory.assert_called_once_with(cwd=tmp_path)
    runtime_factory.assert_called_once_with(str(settings.persist_directory))
    service.index_request.assert_called_once_with(
        request,
        vectorstore=vectorstore,
        settings=settings,
    )


def test_indexing_engine_rejects_unknown_paths(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.md"
    request = IndexDocumentsRequest(document_path=missing_path, cwd=tmp_path)
    engine = IndexingEngine(
        service=Mock(),
        runtime_factory=Mock(),
        settings_factory=Mock(),
    )

    with pytest.raises(InvalidDocumentPathError, match=str(missing_path.resolve())):
        engine.index(request)
