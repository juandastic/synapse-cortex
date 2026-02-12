"""
In-memory job store for async ingest tracking.

Lightweight state only (~200 bytes per job). Compilation is fetched from Neo4j
on-demand when status is "completed", then the job entry is cleaned from memory.
"""

from dataclasses import dataclass
from typing import Literal

_jobs: dict[str, "JobEntry"] = {}


@dataclass
class JobEntry:
    """Lightweight job state for async ingest tracking."""

    status: Literal["processing", "completed", "failed"]
    user_id: str
    session_id: str
    created_at: float
    # Lightweight metadata (no compilation)
    model: str | None = None
    processing_time_ms: float | None = None
    nodes_extracted: int | None = None
    edges_extracted: int | None = None
    episode_id: str | None = None
    error: str | None = None
    code: str | None = None


def create_job(job_id: str, user_id: str, session_id: str) -> bool:
    """
    Create a new job entry. Returns False if job already exists (idempotency guard).
    """
    if job_id in _jobs:
        return False

    import time

    _jobs[job_id] = JobEntry(
        status="processing",
        user_id=user_id,
        session_id=session_id,
        created_at=time.time(),
    )
    return True


def get_job(job_id: str) -> JobEntry | None:
    """Get job entry by ID, or None if not found."""
    return _jobs.get(job_id)


def complete_job(
    job_id: str,
    *,
    model: str | None = None,
    processing_time_ms: float | None = None,
    nodes_extracted: int | None = None,
    edges_extracted: int | None = None,
    episode_id: str | None = None,
) -> None:
    """Mark job as completed with metadata."""
    if job := _jobs.get(job_id):
        job.status = "completed"
        job.model = model
        job.processing_time_ms = processing_time_ms
        job.nodes_extracted = nodes_extracted
        job.edges_extracted = edges_extracted
        job.episode_id = episode_id


def fail_job(job_id: str, error: str, code: str | None = None) -> None:
    """Mark job as failed with error details."""
    if job := _jobs.get(job_id):
        job.status = "failed"
        job.error = error
        job.code = code


def remove_job(job_id: str) -> None:
    """Remove job from dict (called after GET returns terminal state)."""
    _jobs.pop(job_id, None)
