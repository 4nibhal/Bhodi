"""Unit tests for health check endpoint with real services map."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bhodi_platform.application.models import HealthStatus
from bhodi_platform.interfaces.api.app import create_app


@pytest.fixture
def mock_bhodi_app():
    """Provide a mocked BhodiApplication."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def client(mock_bhodi_app):
    """Create a TestClient with a mocked BhodiApplication."""
    with patch("bhodi_platform.interfaces.api.app.Container") as MockContainer:
        mock_container = MockContainer.return_value
        mock_container.build.return_value = mock_bhodi_app
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Tests for GET /health with real health_check behavior."""

    def test_health_returns_200_with_services(self, client, mock_bhodi_app):
        """Health check should return healthy status with services map."""
        mock_bhodi_app.health_check = AsyncMock(
            return_value=HealthStatus(
                status="healthy",
                version="1.0.0",
                services={"embedding": True, "vector_store": True, "llm": True},
            )
        )
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["services"] == {
            "embedding": True,
            "vector_store": True,
            "llm": True,
        }

    def test_health_degraded_returns_503(self, client, mock_bhodi_app):
        """Degraded health check should return 503."""
        mock_bhodi_app.health_check = AsyncMock(
            return_value=HealthStatus(
                status="degraded",
                version="1.0.0",
                services={"embedding": False, "vector_store": True, "llm": True},
            )
        )
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["embedding"] is False
