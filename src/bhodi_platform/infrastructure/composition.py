from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bhodi_platform.answering.application import AnsweringService
from bhodi_platform.answering.application import (
    build_default_answering_service as build_legacy_answering_service,
)
from bhodi_platform.answering.engine import AnsweringEngine
from bhodi_platform.answering.memory import DualVectorStoreConversationMemory
from bhodi_platform.answering.ports import LogWriter as LegacyLogWriter
from bhodi_platform.answering.runtime import get_vectorstore
from bhodi_platform.conversation.runtime import (
    get_persistent_vectorstore as get_persistent_conversation_vectorstore,
)
from bhodi_platform.conversation.runtime import (
    stop_persistent_runtime as stop_conversation_runtime,
)
from bhodi_platform.application.answer_query import AnswerQueryUseCase
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.application.index_documents import IndexDocumentsUseCase
from bhodi_platform.application.runtime import BhodiRuntime
from bhodi_platform.indexing.engine import IndexingEngine
from bhodi_platform.indexing.application import DocumentIndexingService
from bhodi_platform.indexing.errors import InvalidDocumentPathError
from bhodi_platform.indexing.runtime import (
    initialize_persistent_runtime,
    get_persistent_vectorstore,
    stop_persistent_runtime,
)
from bhodi_platform.indexing.settings import IndexingSettings
from bhodi_platform.answering.runtime import stop_runtime as stop_answering_runtime
from bhodi_platform.retrieval.runtime import stop_embeddings


def _noop_log_writer(log_text: str) -> None:
    del log_text
    return None


def build_application(
    log_writer: LegacyLogWriter | None = None,
    *,
    answering_service_factory: Callable[[], AnsweringService] | None = None,
    indexing_service_factory: Callable[[], DocumentIndexingService] | None = None,
    volatile_vectorstore_factory: Callable[[], Any] = get_vectorstore,
    persistent_vectorstore_factory: Callable[
        [], Any
    ] = get_persistent_conversation_vectorstore,
    persistent_runtime_factory: Callable[
        [str], tuple[Any, Any]
    ] = initialize_persistent_runtime,
    indexing_settings_factory: Callable[
        ..., IndexingSettings
    ] = IndexingSettings.from_environment,
) -> BhodiApplication:
    resolved_log_writer: LegacyLogWriter = log_writer or _noop_log_writer

    def conversation_memory_provider() -> DualVectorStoreConversationMemory:
        return DualVectorStoreConversationMemory(
            volatile_vectorstore_factory=volatile_vectorstore_factory,
            persistent_vectorstore_factory=persistent_vectorstore_factory,
        )

    def answer_query_use_case_provider() -> AnswerQueryUseCase:
        service = (
            answering_service_factory()
            if answering_service_factory is not None
            else build_legacy_answering_service(resolved_log_writer)
        )
        return AnswerQueryUseCase(
            engine=AnsweringEngine(service),
            conversation_memory=conversation_memory_provider(),
        )

    def index_documents_use_case_provider() -> IndexDocumentsUseCase:
        service = (
            indexing_service_factory()
            if indexing_service_factory is not None
            else DocumentIndexingService()
        )
        return IndexDocumentsUseCase(
            engine=IndexingEngine(
                service=service,
                runtime_factory=persistent_runtime_factory,
                settings_factory=indexing_settings_factory,
            )
        )

    return BhodiApplication(
        answer_query_use_case_provider=answer_query_use_case_provider,
        index_documents_use_case_provider=index_documents_use_case_provider,
        conversation_memory_provider=conversation_memory_provider,
    )


def build_runtime(
    log_writer: LegacyLogWriter | None = None,
    *,
    answering_service_factory: Callable[[], AnsweringService] | None = None,
    indexing_service_factory: Callable[[], DocumentIndexingService] | None = None,
    volatile_vectorstore_factory: Callable[[], Any] = get_vectorstore,
    persistent_vectorstore_factory: Callable[[], Any] = get_persistent_vectorstore,
    persistent_runtime_factory: Callable[
        [str], tuple[Any, Any]
    ] = initialize_persistent_runtime,
    indexing_settings_factory: Callable[
        ..., IndexingSettings
    ] = IndexingSettings.from_environment,
    shutdown_callbacks: tuple[Callable[[], None], ...] = (
        stop_answering_runtime,
        stop_embeddings,
        stop_conversation_runtime,
        stop_persistent_runtime,
    ),
) -> BhodiRuntime:
    return BhodiRuntime(
        application_factory=lambda: build_application(
            log_writer,
            answering_service_factory=answering_service_factory,
            indexing_service_factory=indexing_service_factory,
            volatile_vectorstore_factory=volatile_vectorstore_factory,
            persistent_vectorstore_factory=persistent_vectorstore_factory,
            persistent_runtime_factory=persistent_runtime_factory,
            indexing_settings_factory=indexing_settings_factory,
        ),
        shutdown_callbacks=shutdown_callbacks,
    )
