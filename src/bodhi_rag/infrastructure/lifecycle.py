from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Generic, Protocol, TypeVar

ResourceT = TypeVar("ResourceT")


class LifecyclePort(Protocol):
    def start(self) -> None:
        """Initialize managed resources."""

    def stop(self) -> None:
        """Release managed resources."""


class ManagedResource(Generic[ResourceT]):
    def __init__(
        self,
        factory: Callable[[], ResourceT],
        *,
        shutdown: Callable[[ResourceT], None] | None = None,
    ) -> None:
        self._factory = factory
        self._shutdown = shutdown
        self._lock = Lock()
        self._resource: ResourceT | None = None

    def start(self) -> ResourceT:
        return self.get()

    def get(self) -> ResourceT:
        if self._resource is not None:
            return self._resource

        with self._lock:
            if self._resource is None:
                self._resource = self._factory()
            return self._resource

    def stop(self) -> None:
        with self._lock:
            resource = self._resource
            self._resource = None

        if resource is not None and self._shutdown is not None:
            self._shutdown(resource)

    def set(self, resource: ResourceT) -> ResourceT:
        with self._lock:
            self._resource = resource
            return resource
