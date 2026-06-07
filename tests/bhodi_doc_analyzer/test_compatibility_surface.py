# ruff: noqa: INP001

import importlib
import sys

import pytest


REMOVED_SUBMODULES = (
    "main",
    "assistant",
    "utils",
    "workflow",
    "chat_service",
)


def _clear_module(name: str) -> None:
    sys.modules.pop(name, None)


@pytest.mark.parametrize("submodule", REMOVED_SUBMODULES)
def test_removed_submodule_imports_raise_module_not_found_error(submodule: str) -> None:
    module_name = f"bhodi_doc_analyzer.{submodule}"
    _clear_module(module_name)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module_name)

    assert excinfo.value.name == module_name


def test_top_level_package_import_still_works() -> None:
    _clear_module("bhodi_doc_analyzer")

    package = importlib.import_module("bhodi_doc_analyzer")

    if package.__name__ != "bhodi_doc_analyzer":
        pytest.fail(f"Expected package name 'bhodi_doc_analyzer', got {package.__name__!r}")
