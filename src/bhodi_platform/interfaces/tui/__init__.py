"""TUI adapters for Bhodi platform workflows."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_OPTIONAL_TUI_MESSAGE = (
    "The `bhodi` TUI requires the optional `tui` extra. "
    "Install it with `uv sync --extra tui` or `uv sync --no-dev --extra tui`."
)

__all__ = [
    "ChatApp",
    "FocusableContainer",
    "MessageBox",
    "main_menu",
    "persist_conversation_turn",
]


def _load_chat_module() -> ModuleType:
    try:
        return import_module("bhodi_platform.interfaces.tui.chat")
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "textual":
            raise SystemExit(_OPTIONAL_TUI_MESSAGE) from error
        raise


def __getattr__(name: str):
    if name in __all__:
        return getattr(_load_chat_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
