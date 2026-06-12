from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from bodhi_rag.infrastructure.lifecycle import ManagedResource

if TYPE_CHECKING:
    from collections.abc import Callable

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
