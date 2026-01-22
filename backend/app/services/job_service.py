"""Database-backed job service.

Provides functions for creating and querying processing jobs in the database.
Used by the API to submit jobs that will be picked up by the worker.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.processing_job import JobStatus, JobType, ProcessingJob


async def create_detection_job(
    session: AsyncSession,
    video_id: int,
    parameters: dict[str, Any] | None = None,
) -> ProcessingJob:
    """Create a new video detection job.

    Args:
        session: Database session.
        video_id: ID of the video to process.
        parameters: Optional job parameters (sample_interval, etc.).

    Returns:
        The created ProcessingJob.
    """
    job = ProcessingJob(
        id=str(uuid.uuid4()),
        job_type=JobType.VIDEO_DETECTION,
        status=JobStatus.PENDING,
        video_id=video_id,
        parameters=parameters or {},
        created_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.flush()  # Get the ID without committing
    return job


async def get_job(session: AsyncSession, job_id: str) -> ProcessingJob | None:
    """Get a job by ID.

    Args:
        session: Database session.
        job_id: Job ID to look up.

    Returns:
        ProcessingJob or None if not found.
    """
    result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_jobs_for_video(
    session: AsyncSession,
    video_id: int,
    status: JobStatus | None = None,
) -> list[ProcessingJob]:
    """Get all jobs for a video.

    Args:
        session: Database session.
        video_id: Video ID.
        status: Optional status filter.

    Returns:
        List of ProcessingJob instances.
    """
    stmt = select(ProcessingJob).where(ProcessingJob.video_id == video_id)
    if status is not None:
        stmt = stmt.where(ProcessingJob.status == status)
    stmt = stmt.order_by(ProcessingJob.created_at.desc())

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def cancel_job(session: AsyncSession, job_id: str) -> bool:
    """Cancel a pending job.

    Only jobs in PENDING status can be cancelled. Jobs that are already
    processing cannot be cancelled (the worker would need to be stopped).

    Args:
        session: Database session.
        job_id: Job ID to cancel.

    Returns:
        True if job was cancelled, False if not found or not cancellable.
    """
    job = await get_job(session, job_id)
    if job is None:
        return False

    if job.status != JobStatus.PENDING:
        return False

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc)
    return True
