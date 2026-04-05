from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bhodi_platform.application.models import (
    IndexDocumentsRequest,
    IndexDocumentsResponse,
)
from bhodi_platform.indexing.application import DocumentIndexingService
from bhodi_platform.indexing.errors import InvalidDocumentPathError
from bhodi_platform.indexing.settings import IndexingSettings


class IndexingEngine:
    def __init__(
        self,
        *,
        service: DocumentIndexingService,
        runtime_factory: Callable[[str], tuple[Any, Any]],
        settings_factory: Callable[..., IndexingSettings],
    ) -> None:
        self._service = service
        self._runtime_factory = runtime_factory
        self._settings_factory = settings_factory

    def index(self, request: IndexDocumentsRequest) -> IndexDocumentsResponse:
        resolved_path = request.document_path.resolve()
        if not resolved_path.is_dir() and not resolved_path.is_file():
            raise InvalidDocumentPathError(str(resolved_path))

        settings = self._settings_factory(cwd=request.cwd or Path.cwd())
        vectorstore, _ = self._runtime_factory(str(settings.persist_directory))
        if hasattr(self._service, "index_request"):
            return self._service.index_request(
                request, vectorstore=vectorstore, settings=settings
            )

        if resolved_path.is_dir():
            indexed_fragments = self._service.index_directory(
                str(resolved_path), vectorstore, settings
            )
            return IndexDocumentsResponse(
                indexed_fragments=indexed_fragments,
                source_kind="directory",
                resolved_path=resolved_path,
            )

        if not resolved_path.is_file():
            raise InvalidDocumentPathError(str(resolved_path))

        indexed_fragments = self._service.index_file(
            str(resolved_path), vectorstore, settings
        )
        return IndexDocumentsResponse(
            indexed_fragments=indexed_fragments,
            source_kind="file",
            resolved_path=resolved_path,
        )
