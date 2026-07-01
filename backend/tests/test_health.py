"""Tests for health check endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use asyncio backend.

    Returns:
        str: Backend identifier for pytest-asyncio.
    """
    return "asyncio"


@pytest.mark.asyncio
async def test_health_endpoint_returns_healthy() -> None:
    """Verify GET /health returns status healthy with HTTP 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_health_endpoint_response_schema() -> None:
    """Verify health response conforms to expected schema fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    payload = response.json()
    assert "status" in payload
    assert isinstance(payload["status"], str)
