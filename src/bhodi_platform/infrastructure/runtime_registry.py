from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from bhodi_platform.infrastructure.lifecycle import ManagedResource

RuntimeT = TypeVar("RuntimeT")


class RuntimeRegistry(ManagedResource[RuntimeT], Generic[RuntimeT]):
    def __init__(self, factory: Callable[[], RuntimeT]) -> None:
        super().__init__(factory)

    def get(self) -> RuntimeT:
        return super().get()

    def start(self) -> RuntimeT:
        return super().start()

    def reset(self) -> None:
        self.stop()

    def stop(self) -> None:
        super().stop()

    def set(self, runtime: RuntimeT) -> RuntimeT:
        return super().set(runtime)
