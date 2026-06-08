__all__ = ["RuntimeRegistry"]


def __getattr__(name: str):
    if name == "RuntimeRegistry":
        from bhodi_platform.infrastructure.runtime_registry import RuntimeRegistry

        return RuntimeRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
