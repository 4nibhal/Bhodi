"""Health check endpoint."""

from fastapi import APIRouter

from bhodi_platform.application.models import HealthStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """
    Health check endpoint.

    Returns status without initializing any models.
    """
    from bhodi_platform.interfaces.api.server import get_bhodi_app

    app = get_bhodi_app()
    return await app.health_check()
