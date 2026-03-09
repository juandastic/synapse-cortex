"""
In-memory job store for Notion export tracking.

Tracks pipeline progress across steps so the polling endpoint can report
which phase the export is in (hydrating, analyzing, creating databases, etc.).
Job entries are removed from memory after the client reads a terminal state.
"""

import time
from dataclasses import dataclass, field
from typing import Literal


_jobs: dict[str, "NotionExportJobEntry"] = {}


@dataclass
class NotionExportJobEntry:
    status: Literal["processing", "completed", "failed"]
    user_id: str
    page_name: str
    current_step: str
    created_at: float
    database_ids: dict[str, str] | None = None
    summary_page_url: str | None = None
    categories_count: int | None = None
    entries_count: int | None = None
    duration_ms: float | None = None
    error: str | None = None
    code: str | None = None


def create_notion_export_job(
    job_id: str,
    user_id: str,
    page_name: str,
) -> bool:
    """Create a new export job. Returns False if job already exists (idempotency guard)."""
    if job_id in _jobs:
        return False

    _jobs[job_id] = NotionExportJobEntry(
        status="processing",
        user_id=user_id,
        page_name=page_name,
        current_step="hydrating",
        created_at=time.time(),
    )
    return True


def get_notion_export_job(job_id: str) -> NotionExportJobEntry | None:
    return _jobs.get(job_id)


def update_notion_export_step(
    job_id: str,
    step: str,
    *,
    categories_count: int | None = None,
    entries_count: int | None = None,
) -> None:
    """Advance the current pipeline step and optionally update progress counters."""
    if job := _jobs.get(job_id):
        job.current_step = step
        if categories_count is not None:
            job.categories_count = categories_count
        if entries_count is not None:
            job.entries_count = entries_count


def complete_notion_export_job(
    job_id: str,
    *,
    database_ids: dict[str, str],
    summary_page_url: str | None = None,
    categories_count: int,
    entries_count: int,
    duration_ms: float,
) -> None:
    if job := _jobs.get(job_id):
        job.status = "completed"
        job.current_step = "done"
        job.database_ids = database_ids
        job.summary_page_url = summary_page_url
        job.categories_count = categories_count
        job.entries_count = entries_count
        job.duration_ms = duration_ms


def fail_notion_export_job(
    job_id: str,
    error: str,
    code: str | None = None,
) -> None:
    if job := _jobs.get(job_id):
        job.status = "failed"
        job.error = error
        job.code = code


def remove_notion_export_job(job_id: str) -> None:
    """Remove job from memory (called after client reads a terminal state)."""
    _jobs.pop(job_id, None)
