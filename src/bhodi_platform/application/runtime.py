from __future__ import annotations

from collections.abc import Callable, Sequence

from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.infrastructure.lifecycle import LifecyclePort


class BhodiRuntime(LifecyclePort):
    def __init__(
        self,
        *,
        application_factory: Callable[[], BhodiApplication],
        shutdown_callbacks: Sequence[Callable[[], None]] = (),
    ) -> None:
        self._application_factory = application_factory
        self._shutdown_callbacks = tuple(shutdown_callbacks)
        self._application: BhodiApplication | None = None
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        for callback in self._shutdown_callbacks:
            callback()
        self._application = None
        self._started = False

    def get_application(self) -> BhodiApplication:
        if self._application is None:
            self._application = self._application_factory()
        return self._application

    def health(self) -> dict[str, bool]:
        return {
            "started": self._started,
            "application_initialized": self._application is not None,
        }
