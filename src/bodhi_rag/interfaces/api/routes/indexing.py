"""Document indexing endpoint."""

from __future__ import annotations

from pathlib import Path as FilePath
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status
from fastapi import Path as PathParam

from bodhi_rag.application.models import (
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from bodhi_rag.domain.exceptions import DocumentNotFoundError

if TYPE_CHECKING:
    from bodhi_rag.interfaces.api.app import ApiSourcePolicy

router = APIRouter()


def _validate_api_source(
    source: str | FilePath,
    policy: ApiSourcePolicy,
) -> FilePath:
    """Validate and resolve an API-local file source."""
    if policy.root is None:
        msg = "Local file indexing via API is disabled until BODHI_API_SOURCE_ROOT is configured"
        raise ValueError(
            msg,
        )

    candidate = FilePath(source).expanduser()
    resolved_path = (
        candidate.resolve() if candidate.is_absolute() else (policy.root / candidate).resolve()
    )

    try:
        resolved_path.relative_to(policy.root)
    except ValueError as exc:
        msg = "Document source must stay within BODHI_API_SOURCE_ROOT"
        raise ValueError(msg) from exc

    if resolved_path.suffix.lower() not in policy.allowed_suffixes:
        allowed_suffixes = ", ".join(sorted(policy.allowed_suffixes))
        msg = f"Document source must use one of: {allowed_suffixes}"
        raise ValueError(msg)

    if not resolved_path.exists():
        msg = "Document source does not exist"
        raise ValueError(msg)

    if not resolved_path.is_file():
        msg = "Document source must be a file"
        raise ValueError(msg)

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
    from bodhi_rag.interfaces.api.app import get_api_source_policy, get_bodhi_rag_app

    app = get_bodhi_rag_app()
    try:
        source_policy = get_api_source_policy()
        resolved_source = _validate_api_source(request.source, source_policy)
        safe_request = request.model_copy(update={"source": resolved_source})
        return await app.index_document(safe_request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str = PathParam(description="Document ID to delete"),
) -> dict:
    """
    Delete a document from the index.

    Removes all chunks associated with the document.
    """
    from bodhi_rag.domain.value_objects import DocumentId
    from bodhi_rag.interfaces.api.app import get_bodhi_rag_app

    app = get_bodhi_rag_app()
    try:
        doc_id = DocumentId(document_id)
        await app.delete_document(doc_id)
        return {"deleted": True, "document_id": document_id}  # noqa: TRY300  # return inside try: the call can raise and we want the except clauses to catch it
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err
