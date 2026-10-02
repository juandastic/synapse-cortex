import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.schemas.models import IngestRequest
from app.services.ingestion import IngestionService
from app.services.job_store import create_job, get_job, remove_job


@pytest.fixture
def ingest_request():
    job_id = f"test-{uuid4()}"
    ingest_request = IngestRequest(
        jobId=job_id,
        userId="owner",
        sessionId="session",
        messages=[{"role": "user", "content": "I live in Bogota", "timestamp": 1000}],
        metadata={"sessionStartedAt": 1000, "sessionEndedAt": 2000, "messageCount": 1},
    )
    yield ingest_request
    remove_job(job_id)


async def test_duplicate_ingest_starts_only_one_background_job(ingest_request):
    service = IngestionService(SimpleNamespace(), "test-model")
    service._process_background = AsyncMock()
    first = await service.accept_session(ingest_request)
    second = await service.accept_session(ingest_request)
    await asyncio.sleep(0)
    assert first.status == second.status == "processing"
    service._process_background.assert_awaited_once_with(ingest_request.jobId, ingest_request)
    assert get_job(ingest_request.jobId).user_id == "owner"


@pytest.mark.parametrize("contents", [[], ["1234"]])
async def test_insufficient_sessions_do_not_create_jobs(ingest_request, contents):
    ingest_request.messages = [
        ingest_request.messages[0].model_copy(update={"content": content}) for content in contents
    ]
    service = IngestionService(SimpleNamespace(), "test-model")
    service._process_background = AsyncMock()
    response = await service.accept_session(ingest_request)
    assert response.status == "skipped"
    assert get_job(ingest_request.jobId) is None
    service._process_background.assert_not_awaited()


async def test_episode_preserves_user_details_and_limits_assistant_noise(ingest_request):
    long_user = "My personal details " * 30
    ingest_request.messages[0].content = long_user
    ingest_request.messages.append(
        ingest_request.messages[0].model_copy(
            update={"role": "assistant", "content": "verbose " * 100}
        )
    )
    graph = SimpleNamespace(
        add_episode=AsyncMock(
            return_value=SimpleNamespace(
                nodes=[object()],
                edges=[object(), object()],
                episode=SimpleNamespace(uuid="episode"),
            )
        )
    )
    service = IngestionService(graph, "test-model")
    service._summarize_episode = AsyncMock()
    create_job(ingest_request.jobId, ingest_request.userId, ingest_request.sessionId)
    await service._process_background(ingest_request.jobId, ingest_request)
    args = graph.add_episode.await_args.kwargs
    assert args["group_id"] == "owner"
    assert long_user in args["episode_body"]
    assert "verbose " * 100 not in args["episode_body"]
    assert args["episode_body"].endswith("...")
    job = get_job(ingest_request.jobId)
    assert job.status == "completed"
    assert job.nodes_extracted == 1
    assert job.edges_extracted == 2
    assert job.episode_id == "episode"


async def test_background_failure_is_pollable_instead_of_leaving_job_processing(ingest_request):
    service = IngestionService(
        SimpleNamespace(add_episode=AsyncMock(side_effect=RuntimeError("graph offline"))),
        "test-model",
    )
    create_job(ingest_request.jobId, ingest_request.userId, ingest_request.sessionId)
    await service._process_background(ingest_request.jobId, ingest_request)
    job = get_job(ingest_request.jobId)
    assert job.status == "failed"
    assert "graph offline" in job.error
