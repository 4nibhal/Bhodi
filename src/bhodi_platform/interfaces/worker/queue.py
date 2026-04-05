from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from bhodi_platform.interfaces.worker.adapter import BhodiWorkerAdapter
from bhodi_platform.interfaces.worker.models import (
    WorkerFailure,
    WorkerJob,
    WorkerRetryPolicy,
    WorkerResult,
)


class JobQueue(Protocol):
    """Protocol for job queue implementations."""

    def submit(self, job: WorkerJob) -> None:
        """Submit a job to the queue."""
        ...

    def start(self) -> None:
        """Start the queue worker thread."""
        ...

    def stop(self, timeout: float | None = None) -> None:
        """Stop the queue worker thread gracefully."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Wait for all jobs to complete before stopping."""
        ...


class JobObserver(Protocol):
    """Protocol for job lifecycle observers."""

    def on_job_submitted(self, job: WorkerJob) -> None:
        """Called when a job is submitted to the queue."""
        ...

    def on_job_started(self, job: WorkerJob, attempt: int) -> None:
        """Called when a job starts processing."""
        ...

    def on_job_succeeded(self, job: WorkerJob, result: WorkerResult) -> None:
        """Called when a job completes successfully."""
        ...

    def on_job_failed(self, job: WorkerJob, result: WorkerResult, attempt: int) -> None:
        """Called when a job fails after all retries."""
        ...


class LoggingJobObserver:
    """Observer that logs job lifecycle events."""

    def on_job_submitted(self, job: WorkerJob) -> None:
        print(f"[job-queue] Job submitted: {job.job_id} ({job.kind})")

    def on_job_started(self, job: WorkerJob, attempt: int) -> None:
        print(f"[job-queue] Job started: {job.job_id} (attempt {attempt})")

    def on_job_succeeded(self, job: WorkerJob, result: WorkerResult) -> None:
        print(f"[job-queue] Job succeeded: {job.job_id}")

    def on_job_failed(self, job: WorkerJob, result: WorkerResult, attempt: int) -> None:
        print(f"[job-queue] Job failed: {job.job_id} after {attempt} attempts")


@dataclass
class _QueueEntry:
    job: WorkerJob
    retry_policy: WorkerRetryPolicy
    observers: tuple[JobObserver, ...]


class InProcessJobQueue:
    """
    In-process job queue that dispatches to BhodiWorkerAdapter.

    Uses a Python queue.Queue internally and a worker thread to process jobs.
    Supports graceful shutdown by draining the queue before stopping.
    """

    def __init__(
        self,
        worker_adapter: BhodiWorkerAdapter,
        retry_policy: WorkerRetryPolicy | None = None,
        observers: list[JobObserver] | None = None,
    ) -> None:
        self._adapter = worker_adapter
        self._retry_policy = retry_policy or WorkerRetryPolicy()
        self._observers: tuple[JobObserver, ...] = tuple(observers or [])
        self._queue: queue.Queue[_QueueEntry | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stopping = False
        self._started = False

    def submit(self, job: WorkerJob) -> None:
        """Submit a job to the queue."""
        entry = _QueueEntry(
            job=job,
            retry_policy=self._retry_policy,
            observers=self._observers,
        )
        self._notify_observers_submit(job)
        self._queue.put(entry)

    def start(self) -> None:
        """Start the queue worker thread."""
        if self._started:
            return
        self._stopping = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._started = True

    def stop(self, timeout: float | None = None) -> None:
        """Stop the queue worker thread gracefully."""
        if not self._started:
            return
        self._stopping = True
        self._queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
        self._started = False

    def join(self, timeout: float | None = None) -> None:
        """Wait for all jobs to complete before stopping."""
        if timeout is not None:
            end_time = time.monotonic() + timeout
            while self._queue.unfinished_tasks > 0:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(0.1, remaining))
        else:
            while self._queue.unfinished_tasks > 0:
                time.sleep(0.1)

    def _worker_loop(self) -> None:
        while True:
            entry = self._queue.get()
            if entry is None:
                self._queue.task_done()
                break
            if self._stopping:
                self._queue.task_done()
                break
            self._process_entry(entry)
            self._queue.task_done()

    def _process_entry(self, entry: _QueueEntry) -> None:
        job = entry.job
        policy = entry.retry_policy
        observers = entry.observers

        attempt = 1
        last_error: str | None = None

        while True:
            self._notify_observers_started(job, attempt)
            result = self._adapter.process(job)

            if result.status == "succeeded":
                self._notify_observers_succeeded(job, result, observers)
                return

            last_error = result.error if hasattr(result, "error") else str(result)
            if attempt >= policy.max_attempts:
                failure_result = WorkerFailure(
                    job_id=job.job_id,
                    kind=job.kind,
                    error=last_error,
                    attempt=attempt,
                )
                self._notify_observers_failed(job, failure_result, attempt, observers)
                return

            delay = policy.next_delay(attempt)
            time.sleep(delay)
            attempt += 1

    def _notify_observers_submit(self, job: WorkerJob) -> None:
        for observer in self._observers:
            observer.on_job_submitted(job)

    def _notify_observers_started(self, job: WorkerJob, attempt: int) -> None:
        for observer in self._observers:
            observer.on_job_started(job, attempt)

    def _notify_observers_succeeded(
        self, job: WorkerJob, result: WorkerResult, observers: tuple[JobObserver, ...]
    ) -> None:
        for observer in observers:
            observer.on_job_succeeded(job, result)

    def _notify_observers_failed(
        self,
        job: WorkerJob,
        result: WorkerFailure,
        attempt: int,
        observers: tuple[JobObserver, ...],
    ) -> None:
        for observer in observers:
            observer.on_job_failed(job, result, attempt)
