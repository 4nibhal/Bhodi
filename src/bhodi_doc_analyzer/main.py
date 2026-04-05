"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from bhodi_platform.interfaces.tui import (
    ChatApp,
    FocusableContainer,
    MessageBox,
    main_menu,
    persist_conversation_turn,
)

__all__ = [
    "ChatApp",
    "FocusableContainer",
    "MessageBox",
    "main_menu",
    "persist_conversation_turn",
]


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main_menu()
