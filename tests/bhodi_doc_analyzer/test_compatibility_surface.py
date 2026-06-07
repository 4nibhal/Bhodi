import importlib
import sys

import pytest


def _clear_module(name: str) -> None:
    sys.modules.pop(name, None)


def test_main_import_currently_fails() -> None:
    _clear_module("bhodi_doc_analyzer.main")

    with pytest.raises((ImportError, AttributeError, SystemExit)):
        importlib.import_module("bhodi_doc_analyzer.main")


def test_assistant_import_currently_fails() -> None:
    _clear_module("bhodi_doc_analyzer.assistant")

    with pytest.raises((ImportError, AttributeError, SystemExit)):
        importlib.import_module("bhodi_doc_analyzer.assistant")


def test_utils_import_currently_fails() -> None:
    _clear_module("bhodi_doc_analyzer.utils")

    with pytest.raises((ImportError, AttributeError)):
        importlib.import_module("bhodi_doc_analyzer.utils")


def test_workflow_import_currently_fails() -> None:
    _clear_module("bhodi_doc_analyzer.workflow")

    with pytest.raises((ImportError, AttributeError, SystemExit)):
        importlib.import_module("bhodi_doc_analyzer.workflow")


def test_chat_service_import_currently_fails() -> None:
    _clear_module("bhodi_doc_analyzer.chat_service")

    with pytest.raises((ImportError, AttributeError, SystemExit)):
        importlib.import_module("bhodi_doc_analyzer.chat_service")


def test_top_level_package_import_still_works() -> None:
    _clear_module("bhodi_doc_analyzer")

    package = importlib.import_module("bhodi_doc_analyzer")

    assert package.__name__ == "bhodi_doc_analyzer"
