"""Health check endpoint."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from bodhi_rag.application.models import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus | JSONResponse:
    """
    Health check endpoint.

    Returns status checking all critical adapters.
    If degraded, returns HTTP 503.
    """
    from bodhi_rag.interfaces.api.app import get_bodhi_rag_app

    app = get_bodhi_rag_app()
    result = await app.health_check()
    if result.status == "degraded":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=result.model_dump(),
        )
    return result
