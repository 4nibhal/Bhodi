"""Document indexing endpoint."""

from fastapi import APIRouter, HTTPException, Path, status

from bhodi_platform.application.models import (
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from bhodi_platform.domain.exceptions import DocumentNotFoundError

router = APIRouter()


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
    from bhodi_platform.interfaces.api.server import get_bhodi_app

    app = get_bhodi_app()
    try:
        return await app.index_document(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str = Path(description="Document ID to delete"),
) -> dict:
    """
    Delete a document from the index.

    Removes all chunks associated with the document.
    """
    from bhodi_platform.interfaces.api.server import get_bhodi_app
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
