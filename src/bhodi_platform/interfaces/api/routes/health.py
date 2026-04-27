"""Health check endpoint."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from bhodi_platform.application.models import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus | JSONResponse:
    """
    Health check endpoint.

    Returns status checking all critical adapters.
    If degraded, returns HTTP 503.
    """
    from bhodi_platform.interfaces.api.app import get_bhodi_app

    app = get_bhodi_app()
    result = await app.health_check()
    if result.status == "degraded":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=result.model_dump(),
        )
    return result
