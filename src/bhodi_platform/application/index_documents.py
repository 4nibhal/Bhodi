from __future__ import annotations

from bhodi_platform.application.models import (
    IndexDocumentsRequest,
    IndexDocumentsResponse,
)
from bhodi_platform.ports import IndexDocumentsEnginePort


class IndexDocumentsUseCase:
    def __init__(self, engine: IndexDocumentsEnginePort) -> None:
        self._engine = engine

    def execute(self, request: IndexDocumentsRequest) -> IndexDocumentsResponse:
        return self._engine.index(request)
