__all__ = [
    "InvalidDocumentPathError",
    "LifecyclePort",
    "ManagedResource",
    "RuntimeRegistry",
]


def __getattr__(name: str):
    if name == "InvalidDocumentPathError":
        from bhodi_platform.indexing.errors import InvalidDocumentPathError

        return InvalidDocumentPathError
    if name == "LifecyclePort" or name == "ManagedResource":
        from bhodi_platform.infrastructure.lifecycle import (
            LifecyclePort,
            ManagedResource,
        )

        if name == "LifecyclePort":
            return LifecyclePort
        return ManagedResource
    if name == "RuntimeRegistry":
        from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry

        return RuntimeRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
