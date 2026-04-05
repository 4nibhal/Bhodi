"""TUI chat interface - DEPRECATED.

This module is retained for backward compatibility only.
New code should use the API or CLI interfaces.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The TUI interface is deprecated and will be removed in a future version. "
    "Use the API or CLI interfaces instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export minimal types for backwards compatibility
from bhodi_platform.application.models import QueryRequest as AnswerQueryRequest
from bhodi_platform.application.models import QueryResponse as ConversationMessage

__all__ = ["AnswerQueryRequest", "ConversationMessage"]
