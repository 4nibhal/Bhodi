from __future__ import annotations

from bhodi_platform.application import (
    AnswerQueryRequest,
    BhodiRuntime,
    IndexDocumentsRequest,
)
from bhodi_platform.interfaces.worker.models import (
    AnswerQueryJob,
    AnswerQueryJobResult,
    IndexDocumentsJob,
    IndexDocumentsJobResult,
    WorkerFailure,
    WorkerJob,
    WorkerResult,
)


class BhodiWorkerAdapter:
    def __init__(self, runtime: BhodiRuntime) -> None:
        self._runtime = runtime

    def start(self) -> None:
        self._runtime.start()

    def stop(self) -> None:
        self._runtime.stop()

    def process(self, job: WorkerJob) -> WorkerResult:
        application = self._runtime.get_application()
        try:
            if isinstance(job, IndexDocumentsJob):
                response = application.index_documents(
                    IndexDocumentsRequest(document_path=job.document_path, cwd=job.cwd)
                )
                return IndexDocumentsJobResult(
                    job_id=job.job_id,
                    indexed_fragments=response.indexed_fragments,
                    source_kind=response.source_kind,
                    resolved_path=response.resolved_path,
                )
            if isinstance(job, AnswerQueryJob):
                response = application.answer_query(
                    AnswerQueryRequest(
                        user_input=job.user_input,
                        messages=job.messages,
                        conversation_id=job.conversation_id,
                    )
                )
                return AnswerQueryJobResult(
                    job_id=job.job_id,
                    answer_text=response.answer_text,
                    context=response.context,
                )
        except Exception as error:
            return WorkerFailure(job_id=job.job_id, kind=job.kind, error=str(error))

        return WorkerFailure(job_id=job.job_id, kind=job.kind, error="Unsupported job")
