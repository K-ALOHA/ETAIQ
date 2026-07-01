"""Health check response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload for the service health endpoint."""

    status: str = Field(
        default="healthy",
        description="Current health status of the API service",
        examples=["healthy"],
    )
