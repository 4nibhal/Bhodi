"""
Deprecated: This module is a compatibility shim that re-exports from bhodi_platform.

All product logic has been moved to src/bhodi_platform/.
This module will be removed in a future release.
"""

from typing import Any

from bhodi_platform.indexing.application import DocumentIndexingService
from bhodi_platform.indexing.settings import IndexingSettings

__all__ = [
    "load_and_index_documents_from_directory",
    "load_and_index_single_file",
]


def _build_indexing_service() -> DocumentIndexingService:
    return DocumentIndexingService()


def load_and_index_documents_from_directory(
    directory_path: str, vectorstore: Any
) -> int:
    """
    Loads documents from a directory, splits them and adds them to the vectorstore.

    Args:
        directory_path (str): Path to the directory containing documents.
        vectorstore (Any): The vectorstore instance where documents will be added.

    Returns:
        int: The number of document fragments indexed.
    """
    service = _build_indexing_service()
    settings = IndexingSettings.from_environment()
    return service.index_directory(directory_path, vectorstore, settings)


def load_and_index_single_file(file_path: str, vectorstore: Any) -> int:
    """
    Loads a single file, splits its contents and adds it to the vectorstore.

    Args:
        file_path (str): Path to the file.
        vectorstore (Any): The vectorstore instance where the document will be added.

    Returns:
        int: The number of document fragments indexed.
    """
    service = _build_indexing_service()
    settings = IndexingSettings.from_environment()
    return service.index_file(file_path, vectorstore, settings)


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
