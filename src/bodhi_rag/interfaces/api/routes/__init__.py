"""API routes for bodhi-rag."""

from bodhi_rag.interfaces.api.routes import health, indexing, query

__all__ = ["health", "indexing", "query"]
