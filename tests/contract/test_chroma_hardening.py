"""
Hardening contract test for CVE-2026-45829 (client-side variant).

CVE-2026-45829 has two components:
  - server-side: in `chromadb/server/fastapi/__init__.py`, reachable only
    when the client uses `chromadb.HttpClient` against a remote server.
    bodhi-rag uses `chromadb.PersistentClient` (embedded mode) so this path
    is unreachable. The `chromadb/chroma` server image is no longer
    pulled by `podman-compose.yml` either.
  - client-side: in `chromadb/api/models/CollectionCommon.py:_embed`,
    reachable only when a collection method is called WITHOUT
    pre-computed embeddings (`embeddings=` or `query_embeddings=`
    set to None or absent). The `_embed` method then falls through to
    `self.configuration.get("embedding_function")`, which calls
    `load_collection_configuration_from_json` and instantiates the
    embedding function from the stored config. A poisoned
    `configuration_json` in the local SQLite db can hijack this.

bodhi-rag's chroma adapter must always supply pre-computed embeddings to
`collection.add`, `query`, `update`, and `upsert`. This test asserts
that structural property of the source so that any future regression
is caught at PR time, not at security-audit time.

Reference: VERSIONS.md and the comment in pyproject.toml at
`chromadb==1.5.9`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


CHROMA_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bodhi_rag"
    / "infrastructure"
    / "vector_store"
    / "chroma.py"
)

# Methods on a chromadb collection that, when called WITHOUT pre-computed
# embeddings, trigger the client-side deserialization sink at
# `chromadb/api/collection_configuration.py:73-91`.
METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS: dict[str, str] = {
    "add": "embeddings",
    "query": "query_embeddings",
    "update": "embeddings",
    "upsert": "embeddings",
}


def _iter_collection_method_calls(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    """Yield (method_name, Call node) for every invocation of a `_collection` method.

    Handles two equivalent patterns used in the adapter:

    1. Direct call: ``self._collection.<method>(...)`` -- the Call node IS
       the call to the collection method and its kwargs are the kwargs
       that reach the chromadb API.

    2. Function reference: ``asyncio.to_thread(self._collection.<method>, ...)``
       -- the collection method is passed by reference as the first
       positional argument to ``asyncio.to_thread``, and the kwargs of
       the ``to_thread`` call are the kwargs that reach the chromadb API.
       We yield the ``to_thread`` Call node in this case so that
       ``_kwargs_of`` reads the same kwargs that the runtime passes to
       the underlying collection method.
    """
    results: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        # Pattern 1: self._collection.<method>(...)
        if isinstance(func.value, ast.Attribute) and func.value.attr == "_collection":
            if isinstance(func.value.value, ast.Name) and func.value.value.id == "self":
                results.append((func.attr, node))
            continue

        # Pattern 2: asyncio.to_thread(self._collection.<method>, **kwargs)
        if func.attr == "to_thread" and isinstance(func.value, ast.Name) and func.value.id == "asyncio":
            if not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Attribute):
                continue
            if not (
                isinstance(first_arg.value, ast.Attribute)
                and first_arg.value.attr == "_collection"
            ):
                continue
            if not (
                isinstance(first_arg.value.value, ast.Name)
                and first_arg.value.value.id == "self"
            ):
                continue
            results.append((first_arg.attr, node))
    return results


def _kwargs_of(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def test_chroma_adapter_always_passes_precomputed_embeddings() -> None:
    """Every `self._collection.{add,query,update,upsert}(...)` call must pass pre-computed embeddings."""
    assert CHROMA_ADAPTER_PATH.exists(), (
        f"Expected chroma adapter at {CHROMA_ADAPTER_PATH}, but the file is missing."
    )
    source = CHROMA_ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    checked: list[tuple[str, str, int]] = []
    violations: list[str] = []

    for method, call in _iter_collection_method_calls(tree):
        if method not in METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS:
            continue
        kwarg_name = METHODS_REQUIRING_PRECOMPUTED_EMBEDDINGS[method]
        kwargs = _kwargs_of(call)
        line = call.lineno

        if kwarg_name not in kwargs:
            violations.append(
                f"Line {line}: `self._collection.{method}(...)` is missing "
                f"`{kwarg_name}=` kwarg. This would re-activate the client-side "
                f"vulnerability path of CVE-2026-45829 (the call would fall through "
                f"to `CollectionCommon._embed`, which instantiates the embedding "
                f"function from the stored configuration)."
            )
            continue
        if isinstance(kwargs[kwarg_name], ast.Constant) and kwargs[kwarg_name].value is None:
            violations.append(
                f"Line {line}: `self._collection.{method}(..., {kwarg_name}=None, ...)` "
                f"passes None. This would re-activate the client-side "
                f"vulnerability path of CVE-2026-45829."
            )
            continue
        checked.append((method, kwarg_name, line))

    assert not violations, "\n".join(violations)
    assert checked, (
        "No `self._collection.add/query/update/upsert` calls were found in "
        "chroma.py. If the adapter was rewritten, update this test to track the "
        "new call sites or remove it if the vulnerability is no longer relevant."
    )
