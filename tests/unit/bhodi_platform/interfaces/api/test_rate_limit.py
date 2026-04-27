"""Unit tests for rate limiting middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bhodi_platform.application.models import (
    HealthStatus,
    IndexDocumentResponse,
    QueryResponse,
)
from bhodi_platform.interfaces.api.app import create_app


@pytest.fixture
def mock_bhodi_app():
    """Provide a mocked BhodiApplication."""
    mock = AsyncMock()
    mock.health_check = AsyncMock(
        return_value=HealthStatus(status="healthy", version="1.0.0")
    )
    mock.index_document = AsyncMock(
        return_value=IndexDocumentResponse(document_id="doc-123", chunk_count=5)
    )
    mock.query = AsyncMock(
        return_value=QueryResponse(
            answer_text="Test answer",
            citations=[],
            conversation_id=None,
        )
    )
    mock.delete_document = AsyncMock(return_value=None)
    mock.get_conversation_history = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def client(mock_bhodi_app):
    """Create a TestClient with a mocked BhodiApplication."""
    # Reset rate limit state so tests are hermetic
    from bhodi_platform.interfaces.api import app as app_module

    app_module._rate_limit_requests.clear()
    with patch("bhodi_platform.interfaces.api.app.Container") as MockContainer:
        mock_container = MockContainer.return_value
        mock_container.build.return_value = mock_bhodi_app
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


class TestRateLimiting:
    """Tests for in-memory rate limiter (100 req/min per IP)."""

    def test_health_not_rate_limited(self, client):
        """Health endpoint should be exempt from rate limiting."""
        for _ in range(105):
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_returns_429(self, client):
        """After 100 non-health requests, should return 429."""
        payload = {"question": "What is this?", "top_k": 3}
        # Send 100 requests (limit)
        for _ in range(100):
            response = client.post("/query", json=payload)
            assert response.status_code == 200

        # 101st request should be rate limited
        response = client.post("/query", json=payload)
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_rate_limit_resets_after_window(self, client):
        """Rate limit should reset after old timestamps expire."""
        payload = {"question": "What is this?", "top_k": 3}
        # Exhaust limit
        for _ in range(100):
            response = client.post("/query", json=payload)
            assert response.status_code == 200

        response = client.post("/query", json=payload)
        assert response.status_code == 429

        # In a real scenario we'd wait 60s; here we patch time.time
        # to simulate the window passing.  Since the middleware uses
        # time.time() directly, we can reset the in-memory dict via
        # the module attribute if we want a quick hack, but a cleaner
        # approach is to monkeypatch time.time in the app module.
        import time

        from bhodi_platform.interfaces.api import app as app_module

        original_time = time.time
        try:
            app_module.time.time = lambda: original_time() + 120
            # Old timestamps are now > 60s old and should be cleaned
            response = client.post("/query", json=payload)
            assert response.status_code == 200
        finally:
            app_module.time.time = original_time
