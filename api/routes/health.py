"""Health check endpoint."""

from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_components, ComponentManager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(components: ComponentManager = Depends(get_components)):
    """Check system health and component availability."""
    status = components.health_check()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        **status,
    )