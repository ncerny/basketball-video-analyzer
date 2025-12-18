"""Background job management system using asyncio.

Provides a simple, in-memory job queue for processing tasks asynchronously.
Designed for local-first video processing where jobs are relatively short-lived.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine


class JobStatus(str, Enum):
    """Status of a background job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a job."""

    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


@dataclass
class Job:
    """Represents a background job."""

    id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percentage": self.progress.percentage,
                "message": self.progress.message,
            },
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


# Type alias for job worker functions
JobWorker = Callable[[Job, Callable[[int, int, str], None]], Coroutine[Any, Any, Any]]


class JobManager:
    """Manages background job execution.

    Features:
    - Async job submission and execution
    - Progress tracking with callbacks
    - Job cancellation support
    - Configurable concurrency limit
    - In-memory job storage (jobs cleaned up after completion)
    """

    def __init__(self, max_concurrent_jobs: int = 2) -> None:
        """Initialize the job manager.

        Args:
            max_concurrent_jobs: Maximum number of jobs to run concurrently.
        """
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._workers: dict[str, JobWorker] = {}
        self._lock = asyncio.Lock()

    def register_worker(self, job_type: str, worker: JobWorker) -> None:
        """Register a worker function for a job type.

        Args:
            job_type: Type identifier for the job.
            worker: Async function that processes the job.
                    Signature: async def worker(job: Job, update_progress: Callable) -> Any
        """
        self._workers[job_type] = worker

    async def submit_job(
        self,
        job_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a new job for processing.

        Args:
            job_type: Type of job (must have registered worker).
            metadata: Optional metadata to pass to the worker.

        Returns:
            Job ID string.

        Raises:
            ValueError: If no worker registered for job type.
        """
        if job_type not in self._workers:
            raise ValueError(f"No worker registered for job type: {job_type}")

        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type=job_type,
            metadata=metadata or {},
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Start job execution in background
        task = asyncio.create_task(self._execute_job(job))
        self._tasks[job_id] = task

        return job_id

    async def _execute_job(self, job: Job) -> None:
        """Execute a job with semaphore limiting."""
        async with self._semaphore:
            await self._run_job(job)

    async def _run_job(self, job: Job) -> None:
        """Run the job worker function."""
        worker = self._workers[job.job_type]

        def update_progress(current: int, total: int, message: str = "") -> None:
            """Progress callback for workers."""
            job.progress.current = current
            job.progress.total = total
            job.progress.message = message

        try:
            # Mark as processing
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)

            # Run the worker
            result = await worker(job, update_progress)

            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now(timezone.utc)
            job.progress.current = job.progress.total  # Ensure 100%

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            raise

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID.

        Args:
            job_id: Job ID to look up.

        Returns:
            Job instance or None if not found.
        """
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> JobStatus | None:
        """Get job status by ID.

        Args:
            job_id: Job ID to look up.

        Returns:
            JobStatus or None if job not found.
        """
        job = self._jobs.get(job_id)
        return job.status if job else None

    def get_all_jobs(self, job_type: str | None = None) -> list[Job]:
        """Get all jobs, optionally filtered by type.

        Args:
            job_type: Optional filter by job type.

        Returns:
            List of Job instances.
        """
        jobs = list(self._jobs.values())
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        return jobs

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if job was cancelled, False if not found or already complete.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        # Cancel the task if running
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        return True

    async def wait_for_job(self, job_id: str, timeout: float | None = None) -> Job | None:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for.
            timeout: Maximum seconds to wait (None = wait forever).

        Returns:
            Completed Job instance or None if not found/timeout.
        """
        task = self._tasks.get(job_id)
        if not task:
            return self._jobs.get(job_id)

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

        return self._jobs.get(job_id)

    def cleanup_completed_jobs(self, max_age_seconds: float = 3600) -> int:
        """Remove completed jobs older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds for completed jobs.

        Returns:
            Number of jobs removed.
        """
        now = datetime.now(timezone.utc)
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]
            self._tasks.pop(job_id, None)

        return len(to_remove)

    @property
    def active_job_count(self) -> int:
        """Get count of active (pending/processing) jobs."""
        return sum(
            1
            for job in self._jobs.values()
            if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
        )

    @property
    def registered_job_types(self) -> list[str]:
        """Get list of registered job types."""
        return list(self._workers.keys())


# Global job manager instance
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        Global JobManager instance.
    """
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def reset_job_manager() -> None:
    """Reset the global job manager (primarily for testing)."""
    global _job_manager
    _job_manager = None
