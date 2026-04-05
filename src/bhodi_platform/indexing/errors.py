from __future__ import annotations


class InvalidDocumentPathError(ValueError):
    """Raised when a document indexing request points to neither a file nor a directory."""
