from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from importlib import import_module
from types import ModuleType


def install_answering_test_stubs() -> None:
    if "langchain_core.messages" in sys.modules:
        return

    messages = ModuleType("langchain_core.messages")
    output_parsers = ModuleType("langchain_core.output_parsers")
    graph = ModuleType("langgraph.graph")
    textual_app = ModuleType("textual.app")
    textual_binding = ModuleType("textual.binding")
    textual_containers = ModuleType("textual.containers")
    textual_widget = ModuleType("textual.widget")
    textual_widgets = ModuleType("textual.widgets")

    class _BaseMessage:
        def __init__(self, content=None, **kwargs):
            self.content = content if content is not None else kwargs.get("content", "")

    class HumanMessage(_BaseMessage):
        pass

    class AIMessage(_BaseMessage):
        pass

    class PydanticOutputParser:
        def __init__(self, pydantic_object):
            self._model = pydantic_object

        def parse(self, raw_text):
            if not isinstance(raw_text, str):
                raise ValueError("expected string response")
            payload = json.loads(raw_text)
            return self._model(**payload)

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}

        def add_node(self, name, node):
            self.nodes[name] = node

        def set_entry_point(self, _name):
            return None

        def add_edge(self, _source, _target):
            return None

        def compile(self):
            return object()

    class App:
        def __init__(self, *args, **kwargs):
            return None

    class ComposeResult:
        pass

    class Binding:
        def __init__(self, *args, **kwargs):
            return None

    class Container:
        def __init__(self, *args, **kwargs):
            return None

    class Horizontal(Container):
        pass

    class Widget:
        disabled = False

        def __init__(self, *args, **kwargs):
            return None

    class Button(Widget):
        pass

    class Footer(Widget):
        pass

    class Header(Widget):
        pass

    class Input(Widget):
        Changed = object()

        def __init__(self, *args, **kwargs):
            self.value = ""

        def prevent(self, _event):
            return nullcontext()

    class Markdown(Widget):
        pass

    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    output_parsers.PydanticOutputParser = PydanticOutputParser
    graph.END = object()
    graph.StateGraph = StateGraph
    textual_app.App = App
    textual_app.ComposeResult = ComposeResult
    textual_binding.Binding = Binding
    textual_containers.Container = Container
    textual_containers.Horizontal = Horizontal
    textual_widget.Widget = Widget
    textual_widgets.Button = Button
    textual_widgets.Footer = Footer
    textual_widgets.Header = Header
    textual_widgets.Input = Input
    textual_widgets.Markdown = Markdown

    sys.modules["langchain_core"] = ModuleType("langchain_core")
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_core.output_parsers"] = output_parsers
    import_module("pydantic")
    sys.modules["langgraph"] = ModuleType("langgraph")
    sys.modules["langgraph.graph"] = graph
    sys.modules["textual"] = ModuleType("textual")
    sys.modules["textual.app"] = textual_app
    sys.modules["textual.binding"] = textual_binding
    sys.modules["textual.containers"] = textual_containers
    sys.modules["textual.widget"] = textual_widget
    sys.modules["textual.widgets"] = textual_widgets
