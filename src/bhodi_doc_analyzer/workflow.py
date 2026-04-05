"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from bhodi_platform.answering.application import (
    AgentState,
    AnsweringService,
    answer_parser,
    build_default_answering_service,
)
from bhodi_platform.interfaces.tui.chat import _save_log as save_log

__all__ = [
    "AgentState",
    "CompiledWorkflow",
    "answer_parser",
    "compiled_graph",
    "execute_workflow",
    "generate_response",
    "graph",
    "refine_prompt",
    "rerank_documents",
    "retrieve_context",
    "sequence_documents",
    "summarize_text",
    "transform_message",
    "workflow",
]


def _get_answering_service() -> AnsweringService:
    return build_default_answering_service(save_log)


def retrieve_context(state: AgentState):
    return _get_answering_service().retrieve_context(state)


def generate_response(state: AgentState):
    return _get_answering_service().generate_response(state)


def summarize_text(text: str) -> str:
    return _get_answering_service().summarize_text(text)


def refine_prompt(context: str, user_input: str) -> str:
    return _get_answering_service().refine_prompt(context, user_input)


def transform_message(msg: dict):
    return _get_answering_service().transform_message(msg)


def rerank_documents(query: str, docs: list):
    return _get_answering_service().rerank_documents(query, docs)


def sequence_documents(query: str, docs: list):
    return _get_answering_service().sequence_documents(query, docs)


class CompiledWorkflow:
    def invoke(self, state: AgentState) -> AgentState:
        return execute_workflow(state)


compiled_graph = CompiledWorkflow()
workflow = compiled_graph


def execute_workflow(state: AgentState) -> AgentState:
    return _get_answering_service().execute(state)


class CallableStateGraph:
    def __init__(self, graph, executor) -> None:
        self.graph = graph
        self.executor = executor

    def __call__(self, state: AgentState) -> AgentState:
        return self.executor(state)


graph = CallableStateGraph(compiled_graph, execute_workflow)


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
