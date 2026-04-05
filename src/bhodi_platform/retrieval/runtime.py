from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry
from bhodi_platform.retrieval.settings import EmbeddingSettings


def _embeddings_class() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings


def build_embeddings(model_name: str, device: str) -> Any:
    return _embeddings_class()(
        model_name=model_name,
        model_kwargs={"device": device},
    )


def _build_embeddings_from_environment() -> Any:
    settings = EmbeddingSettings.from_environment()
    return build_embeddings(settings.model, settings.device)


_embeddings_registry = RuntimeRegistry(_build_embeddings_from_environment)


def start_embeddings() -> Any:
    return _embeddings_registry.start()


def get_embeddings() -> Any:
    return _embeddings_registry.get()


def reset_embeddings() -> None:
    _embeddings_registry.reset()


def stop_embeddings() -> None:
    _embeddings_registry.stop()


class LazyObjectProxy:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory()(*args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._factory(), attribute)


embeddings = LazyObjectProxy(get_embeddings)
