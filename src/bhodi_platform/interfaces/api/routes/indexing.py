"""Document indexing endpoint."""

from __future__ import annotations

from pathlib import Path as FilePath
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Path as PathParam, status

from bhodi_platform.application.models import (
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from bhodi_platform.domain.exceptions import DocumentNotFoundError

if TYPE_CHECKING:
    from bhodi_platform.interfaces.api.app import ApiSourcePolicy

router = APIRouter()


def _validate_api_source(
    source: str | FilePath,
    policy: ApiSourcePolicy,
) -> FilePath:
    """Validate and resolve an API-local file source."""
    if policy.root is None:
        raise ValueError(
            "Local file indexing via API is disabled until BHODI_API_SOURCE_ROOT is configured"
        )

    candidate = FilePath(source).expanduser()
    resolved_path = (
        candidate.resolve() if candidate.is_absolute() else (policy.root / candidate).resolve()
    )

    try:
        resolved_path.relative_to(policy.root)
    except ValueError as exc:
        raise ValueError("Document source must stay within BHODI_API_SOURCE_ROOT") from exc

    if resolved_path.suffix.lower() not in policy.allowed_suffixes:
        allowed_suffixes = ", ".join(sorted(policy.allowed_suffixes))
        raise ValueError(f"Document source must use one of: {allowed_suffixes}")

    if not resolved_path.exists():
        raise ValueError("Document source does not exist")

    if not resolved_path.is_file():
        raise ValueError("Document source must be a file")

    return resolved_path


@router.post(
    "/documents",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def index_document(request: IndexDocumentRequest) -> IndexDocumentResponse:
    """
    Index a document.

    Upload and index a document for later querying.
    """
    from bhodi_platform.interfaces.api.app import get_api_source_policy, get_bhodi_app

    app = get_bhodi_app()
    try:
        source_policy = get_api_source_policy()
        resolved_source = _validate_api_source(request.source, source_policy)
        safe_request = request.model_copy(update={"source": resolved_source})
        return await app.index_document(safe_request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str = PathParam(description="Document ID to delete"),
) -> dict:
    """
    Delete a document from the index.

    Removes all chunks associated with the document.
    """
    from bhodi_platform.interfaces.api.app import get_bhodi_app
    from bhodi_platform.domain.value_objects import DocumentId

    app = get_bhodi_app()
    try:
        doc_id = DocumentId(document_id)
        await app.delete_document(doc_id)
        return {"deleted": True, "document_id": document_id}
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
