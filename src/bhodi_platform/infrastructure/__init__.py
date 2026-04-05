__all__ = [
    "InvalidDocumentPathError",
    "LifecyclePort",
    "ManagedResource",
    "RuntimeRegistry",
    "build_application",
    "build_runtime",
]


def __getattr__(name: str):
    if name in {"build_application", "build_runtime", "InvalidDocumentPathError"}:
        from bhodi_platform.infrastructure.composition import (
            InvalidDocumentPathError,
            build_application,
            build_runtime,
        )

        if name == "build_application":
            return build_application
        if name == "build_runtime":
            return build_runtime
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
