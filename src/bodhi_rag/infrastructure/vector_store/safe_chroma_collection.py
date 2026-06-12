"""
Defensive wrapper around a chromadb Collection.

The wrapper is a security perimeter: it restricts the methods the rest of
the codebase can call on a chromadb collection, and it enforces that
`add` / `query` are always called with pre-computed embeddings.

This neutralizes the client-side variant of CVE-2026-45829 (and any
future CVE that shares the "instantiate an embedding function from the
collection's stored configuration" shape) regardless of which
chromadb version we are pinned to, as long as the wrapper is the only
path to the underlying collection.

The wrapper enforces two invariants:

  1. `add` and `query` MUST receive pre-computed embeddings. Passing
     `None` or omitting the kwarg raises immediately. This blocks
     `chromadb/api/models/CollectionCommon.py:_embed` from being
     reached, which is the only call site that loads
     `load_collection_configuration_from_json` and instantiates
     objects from the stored configuration.

  2. The wrapper exposes only the four methods bodhi-rag actually uses
     (`add`, `query`, `get`, `delete`). Any attempt to call other
     chromadb methods (`update`, `upsert`, `peek`, `modify`,
     `_embed_search_string_queries`, etc.) raises `AttributeError`,
     making it impossible to add new attack surfaces by accident.
"""

from __future__ import annotations

from typing import Any


class SafeChromaCollection:
    """Restricts and validates a chromadb Collection."""

    _ALLOWED_METHODS = frozenset({"add", "query", "get", "delete"})

    def __init__(self, raw_collection: Any) -> None:
        # Bind the public methods to self via the class dict so that
        # `getattr(wrapper, "unknown_method")` raises AttributeError
        # rather than exposing the raw collection's full surface.
        self.__dict__["_raw"] = raw_collection

    def __getattr__(self, name: str) -> Any:
        if name in self._ALLOWED_METHODS:
            method = getattr(self.__dict__["_raw"], name)

            if name == "add":
                return self._safe_add(method)
            if name == "query":
                return self._safe_query(method)
            return method
        raise AttributeError(
            f"SafeChromaCollection does not expose '{name}'. "
            f"Allowed methods: {sorted(self._ALLOWED_METHODS)}. "
            f"If you need a new method, add it to _ALLOWED_METHODS "
            f"after a security review of its embedding deserialization "
            f"behavior."
        )

    @staticmethod
    def _safe_add(raw_add: Any):
        def _add(*, ids, embeddings, documents, metadatas):
            if embeddings is None:
                raise ValueError(
                    "SafeChromaCollection.add: `embeddings` must be pre-"
                    "computed and not None. Passing None would let "
                    "chromadb fall through to "
                    "`CollectionCommon._embed`, which instantiates the "
                    "embedding function from the stored configuration "
                    "(client-side variant of CVE-2026-45829)."
                )
            return raw_add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        return _add

    @staticmethod
    def _safe_query(raw_query: Any):
        def _query(*, query_embeddings, n_results):
            if query_embeddings is None:
                raise ValueError(
                    "SafeChromaCollection.query: `query_embeddings` must "
                    "be pre-computed and not None. Passing None would let "
                    "chromadb fall through to "
                    "`CollectionCommon._embed`, which instantiates the "
                    "embedding function from the stored configuration "
                    "(client-side variant of CVE-2026-45829)."
                )
            return raw_query(
                query_embeddings=query_embeddings,
                n_results=n_results,
            )

        return _query
