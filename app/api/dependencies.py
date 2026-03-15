"""
API Dependencies - Dependency injection for database connections and services.
"""

from typing import Annotated

from fastapi import Depends, Request

from graphiti_core import Graphiti

from app.core.security import verify_api_key
from app.services.generation import GenerationService
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.ingestion import IngestionService
from app.services.notion_correction import NotionCorrectionService
from app.services.notion_export import NotionExportService


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


def get_graphiti(request: Request) -> Graphiti:
    """Get the Graphiti client from app state."""
    return request.app.state.graphiti


def get_notion_export_service(request: Request) -> NotionExportService:
    """Get the Notion export service from app state."""
    return request.app.state.notion_export_service


def get_notion_correction_service(request: Request) -> NotionCorrectionService:
    """Get the Notion correction import service from app state."""
    return request.app.state.notion_correction_service


# Type aliases for service dependencies
HydrationServiceDep = Annotated[HydrationService, Depends(get_hydration_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]
GraphServiceDep = Annotated[GraphService, Depends(get_graph_service)]
GraphitiDep = Annotated[Graphiti, Depends(get_graphiti)]
NotionExportServiceDep = Annotated[NotionExportService, Depends(get_notion_export_service)]
NotionCorrectionServiceDep = Annotated[NotionCorrectionService, Depends(get_notion_correction_service)]
