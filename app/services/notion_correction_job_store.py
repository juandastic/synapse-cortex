"""
In-memory job store for Notion correction import tracking.

Tracks pipeline progress across steps so the polling endpoint can report
which phase the import is in (scanning, building corrections, applying).
Job entries are removed from memory after the client reads a terminal state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal


_jobs: dict[str, "NotionCorrectionJobEntry"] = {}


@dataclass
class NotionCorrectionJobEntry:
    status: Literal["processing", "completed", "failed"]
    group_id: str
    current_step: str
    created_at: float
    databases_scanned: int | None = None
    corrections_found: int | None = None
    corrections_applied: int | None = None
    corrections_failed: int | None = None
    failed_corrections: list[dict] | None = None
    duration_ms: float | None = None
    error: str | None = None
    code: str | None = None


def create_notion_correction_job(
    job_id: str,
    group_id: str,
) -> bool:
    """Create a new correction job. Returns False if job already exists (idempotency guard)."""
    if job_id in _jobs:
        return False

    _jobs[job_id] = NotionCorrectionJobEntry(
        status="processing",
        group_id=group_id,
        current_step="scanning",
        created_at=time.time(),
    )
    return True


def get_notion_correction_job(job_id: str) -> NotionCorrectionJobEntry | None:
    return _jobs.get(job_id)


def update_notion_correction_step(
    job_id: str,
    step: str,
    *,
    databases_scanned: int | None = None,
    corrections_found: int | None = None,
    corrections_applied: int | None = None,
    corrections_failed: int | None = None,
) -> None:
    """Advance the current pipeline step and optionally update progress counters."""
    if job := _jobs.get(job_id):
        job.current_step = step
        if databases_scanned is not None:
            job.databases_scanned = databases_scanned
        if corrections_found is not None:
            job.corrections_found = corrections_found
        if corrections_applied is not None:
            job.corrections_applied = corrections_applied
        if corrections_failed is not None:
            job.corrections_failed = corrections_failed


def complete_notion_correction_job(
    job_id: str,
    *,
    corrections_found: int,
    corrections_applied: int,
    corrections_failed: int,
    failed_corrections: list[dict] | None,
    duration_ms: float,
) -> None:
    if job := _jobs.get(job_id):
        job.status = "completed"
        job.current_step = "done"
        job.corrections_found = corrections_found
        job.corrections_applied = corrections_applied
        job.corrections_failed = corrections_failed
        job.failed_corrections = failed_corrections
        job.duration_ms = duration_ms


def fail_notion_correction_job(
    job_id: str,
    error: str,
    code: str | None = None,
) -> None:
    if job := _jobs.get(job_id):
        job.status = "failed"
        job.error = error
        job.code = code


def remove_notion_correction_job(job_id: str) -> None:
    """Remove job from memory (called after client reads a terminal state)."""
    _jobs.pop(job_id, None)
