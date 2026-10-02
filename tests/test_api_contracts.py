from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import router
from app.core import security
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.job_store import complete_job, create_job, fail_job, get_job, remove_job


@pytest.fixture
async def api_client(monkeypatch):
    # Mount the production router without production lifespan: no Neo4j, LLM or telemetry startup.
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(synapse_api_secret="test-secret")
    )
    app.state.hydration_service = SimpleNamespace(
        build_user_knowledge=AsyncMock(
            return_value=HydrationResult(
                compilation_text="User lives in Bogota",
                metadata=CompilationMetadata(
                    is_partial=True, total_estimated_tokens=12, included_node_ids=["node"]
                ),
                graph_stats=GraphStats(entity_count=1, relationship_count=2),
            )
        )
    )
    app.state.cache_manager = SimpleNamespace(
        create_compilation_cache=AsyncMock(return_value=("cachedContents/test", ""))
    )
    app.state.ingestion_service = SimpleNamespace(accept_session=AsyncMock())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        yield client, app.state


@pytest.fixture
def job_id():
    job_id = f"test-{uuid4()}"
    create_job(job_id, "owner", "session")
    yield job_id
    remove_job(job_id)


async def test_health_is_public_but_memory_routes_require_valid_api_secret(api_client):
    client, services = api_client
    assert (await client.get("/health")).status_code == 200
    for headers, code in [({}, 422), ({"X-API-SECRET": "wrong"}, 401)]:
        response = await client.get("/ingest/status/unknown", headers=headers)
        assert response.status_code == code
    services.hydration_service.build_user_knowledge.assert_not_awaited()


async def test_processing_poll_does_not_compile_memory_or_remove_the_job(api_client, job_id):
    client, services = api_client
    response = await client.get(f"/ingest/status/{job_id}", headers={"X-API-SECRET": "test-secret"})
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert get_job(job_id) is not None
    services.hydration_service.build_user_knowledge.assert_not_awaited()
    services.cache_manager.create_compilation_cache.assert_not_awaited()


async def test_completed_poll_returns_memory_cache_and_metadata_then_removes_job(
    api_client, job_id
):
    client, services = api_client
    complete_job(job_id, model="test-model", nodes_extracted=1, edges_extracted=2)
    response = await client.get(
        f"/ingest/status/{job_id}?version=v2", headers={"X-API-SECRET": "test-secret"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["userKnowledgeCompilation"] == "User lives in Bogota"
    assert body["cacheName"] == "cachedContents/test"
    assert body["compilationMetadata"]["included_node_ids"] == ["node"]
    assert body["graphStats"]["relationship_count"] == 2
    assert body["metadata"]["model"] == "test-model"
    services.hydration_service.build_user_knowledge.assert_awaited_once_with("owner", version="v2")
    assert get_job(job_id) is None
    assert (
        await client.get(f"/ingest/status/{job_id}", headers={"X-API-SECRET": "test-secret"})
    ).status_code == 404


async def test_failed_poll_returns_retry_information_and_cleans_job(api_client, job_id):
    client, _ = api_client
    fail_job(job_id, "Provider unavailable", "PROVIDER_ERROR")
    response = await client.get(f"/ingest/status/{job_id}", headers={"X-API-SECRET": "test-secret"})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["code"] == "PROVIDER_ERROR"
    assert get_job(job_id) is None


@pytest.mark.parametrize("service", ["hydration", "cache"])
async def test_poll_failure_preserves_completed_job_so_client_can_retry(
    api_client, job_id, service
):
    client, services = api_client
    complete_job(job_id)
    method = (
        services.hydration_service.build_user_knowledge
        if service == "hydration"
        else services.cache_manager.create_compilation_cache
    )
    method.side_effect = RuntimeError("temporarily unavailable")
    assert (
        await client.get(f"/ingest/status/{job_id}", headers={"X-API-SECRET": "test-secret"})
    ).status_code == 500
    assert get_job(job_id).status == "completed"
    method.side_effect = None
    assert (
        await client.get(f"/ingest/status/{job_id}", headers={"X-API-SECRET": "test-secret"})
    ).status_code == 200
    assert get_job(job_id) is None
