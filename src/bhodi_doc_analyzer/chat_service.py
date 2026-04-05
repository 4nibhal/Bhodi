"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from bhodi_platform.answering.runtime import llm
from bhodi_platform.answering.application import build_default_answering_service
from bhodi_platform.answering.collaborators import answer_parser
from bhodi_platform.interfaces.tui.chat import _save_log as save_log

__all__ = [
    "generate_chat_response",
    "llm",
    "_get_answering_service",
    "answer_parser",
    "save_log",
]


def _get_answering_service():
    return build_default_answering_service(save_log)


def generate_chat_response(tech_prompt: str) -> str:
    """
    Generates a chat response from the language model using the provided technical prompt.
    This function is executed synchronously and is designed to be called from a separate thread.

    Args:
        tech_prompt (str): The technical prompt for the language model.

    Returns:
        str: The generated answer from the language model.
    """
    from langchain_core.messages import HumanMessage

    raw_response: str = llm.invoke([HumanMessage(tech_prompt)])
    save_log(f"Raw response (structured mode): {raw_response}")
    _ = answer_parser
    return _get_answering_service().parse_answer_response(
        raw_response,
        error_prefix="Parsing error (structured mode)",
        coerce_plain_text=False,
    )


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
