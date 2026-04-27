"""Unit tests for Bhodi API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bhodi_platform.application.models import (
    HealthStatus,
    IndexDocumentResponse,
    QueryResponse,
)
from bhodi_platform.domain.exceptions import DocumentNotFoundError
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
    with patch("bhodi_platform.interfaces.api.app.Container") as MockContainer:
        mock_container = MockContainer.return_value
        mock_container.build.return_value = mock_bhodi_app
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_returns_200(self, client):
        """Query endpoint should return answer."""
        payload = {"question": "What is this?", "top_k": 3}
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer_text"] == "Test answer"

    def test_query_invalid_body_returns_422(self, client):
        """Invalid body should trigger validation error."""
        response = client.post("/query", json={})
        assert response.status_code == 422


class TestDocumentsEndpoint:
    """Tests for POST /documents and DELETE /documents/{id}."""

    def test_index_document_returns_201(self, client):
        """Indexing a document should return 201."""
        payload = {"source": "test.pdf", "metadata": {"key": "value"}}
        response = client.post("/documents", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["chunk_count"] == 5

    def test_delete_document_returns_200(self, client):
        """Deleting a document should return confirmation."""
        response = client.delete("/documents/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["document_id"] == "12345678-1234-1234-1234-123456789abc"

    def test_delete_document_not_found_returns_404(self, client, mock_bhodi_app):
        """Deleting a missing document should return 404."""
        mock_bhodi_app.delete_document.side_effect = DocumentNotFoundError(
            "12345678-1234-1234-1234-123456789abc"
        )
        response = client.delete("/documents/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"


class TestConversationsEndpoint:
    """Tests for GET /conversations/{id}."""

    def test_get_conversation_returns_200(self, client, mock_bhodi_app):
        """Getting conversation history should return turns."""
        turn = AsyncMock()
        turn.user_message = "Hello"
        turn.assistant_message = "Hi there"
        turn.turn_index = 0
        mock_bhodi_app.get_conversation_history.return_value = [turn]

        response = client.get("/conversations/12345678-1234-1234-1234-123456789abc")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "12345678-1234-1234-1234-123456789abc"
        assert len(data["turns"]) == 1
        assert data["turns"][0]["user_message"] == "Hello"
