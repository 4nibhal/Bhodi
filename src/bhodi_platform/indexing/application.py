from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bhodi_platform.application.models import (
    IndexDocumentsRequest,
    IndexDocumentsResponse,
)
from bhodi_platform.domain import IndexingDomainService
from bhodi_platform.domain.exceptions import PolicyViolationError
from bhodi_platform.indexing.errors import InvalidDocumentPathError
from bhodi_platform.indexing.infrastructure import (
    build_splitter,
    load_documents_from_directory,
    load_documents_from_file,
)
from bhodi_platform.indexing.ports import DocumentSplitter, VectorStorePort
from bhodi_platform.indexing.settings import IndexingSettings


class DocumentIndexingService:
    def __init__(
        self,
        directory_loader: Callable[[str], list[Any]] = load_documents_from_directory,
        file_loader: Callable[[str], list[Any]] = load_documents_from_file,
        splitter_factory: Callable[
            [IndexingSettings], DocumentSplitter
        ] = build_splitter,
        domain_service: IndexingDomainService | None = None,
    ) -> None:
        self._directory_loader = directory_loader
        self._file_loader = file_loader
        self._splitter_factory = splitter_factory
        self._domain_service = domain_service or IndexingDomainService()

    def index_directory(
        self,
        directory_path: str,
        vectorstore: VectorStorePort,
        settings: IndexingSettings,
    ) -> int:
        documents = self._directory_loader(directory_path)
        return self._split_and_store(documents, vectorstore, settings)

    def index_file(
        self,
        file_path: str,
        vectorstore: VectorStorePort,
        settings: IndexingSettings,
    ) -> int:
        documents = self._file_loader(file_path)
        return self._split_and_store(documents, vectorstore, settings)

    def index_request(
        self,
        request: IndexDocumentsRequest,
        *,
        vectorstore: VectorStorePort,
        settings: IndexingSettings,
    ) -> IndexDocumentsResponse:
        resolved_path = request.document_path.resolve()
        if not resolved_path.is_dir() and not resolved_path.is_file():
            raise InvalidDocumentPathError(str(resolved_path))

        # Delegate path validation to domain policy
        try:
            self._domain_service.validate_index_request(resolved_path)
        except PolicyViolationError as e:
            raise InvalidDocumentPathError(str(e)) from e

        path_text = str(resolved_path)
        if resolved_path.is_dir():
            indexed_fragments = self.index_directory(path_text, vectorstore, settings)
            return IndexDocumentsResponse(
                indexed_fragments=indexed_fragments,
                source_kind="directory",
                resolved_path=resolved_path,
            )

        indexed_fragments = self.index_file(path_text, vectorstore, settings)
        return IndexDocumentsResponse(
            indexed_fragments=indexed_fragments,
            source_kind="file",
            resolved_path=resolved_path,
        )

    def _split_and_store(
        self,
        documents: list[Any],
        vectorstore: VectorStorePort,
        settings: IndexingSettings,
    ) -> int:
        splitter = self._splitter_factory(settings)
        split_documents = splitter.split_documents(documents)
        vectorstore.add_documents(split_documents)
        return len(split_documents)
