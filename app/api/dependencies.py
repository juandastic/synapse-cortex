"""
API Dependencies - Dependency injection for database connections and services.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.security import verify_api_key
from app.services.generation import GenerationService
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.ingestion import IngestionService


# Type alias for API key dependency
ApiKeyDep = Annotated[str, Depends(verify_api_key)]


def get_hydration_service(request: Request) -> HydrationService:
    """Get the hydration service from app state."""
    return request.app.state.hydration_service


def get_ingestion_service(request: Request) -> IngestionService:
    """Get the ingestion service from app state."""
    return request.app.state.ingestion_service


def get_generation_service(request: Request) -> GenerationService:
    """Get the generation service from app state."""
    return request.app.state.generation_service


def get_graph_service(request: Request) -> GraphService:
    """Get the graph service from app state."""
    return request.app.state.graph_service


# Type aliases for service dependencies
HydrationServiceDep = Annotated[HydrationService, Depends(get_hydration_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]
GraphServiceDep = Annotated[GraphService, Depends(get_graph_service)]
