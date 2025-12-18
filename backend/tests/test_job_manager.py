"""Tests for the background job manager."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.services.job_manager import (
    Job,
    JobManager,
    JobProgress,
    JobStatus,
    get_job_manager,
    reset_job_manager,
)


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = JobProgress(current=50, total=100)
        assert progress.percentage == 50.0

    def test_percentage_zero_total(self):
        """Test percentage with zero total returns 0."""
        progress = JobProgress(current=0, total=0)
        assert progress.percentage == 0.0

    def test_percentage_complete(self):
        """Test percentage at 100%."""
        progress = JobProgress(current=100, total=100)
        assert progress.percentage == 100.0


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test creating a job with defaults."""
        job = Job(id="test-1", job_type="test")
        assert job.id == "test-1"
        assert job.job_type == "test"
        assert job.status == JobStatus.PENDING
        assert job.result is None
        assert job.error is None

    def test_job_to_dict(self):
        """Test job serialization to dictionary."""
        job = Job(id="test-1", job_type="test", metadata={"key": "value"})
        data = job.to_dict()

        assert data["id"] == "test-1"
        assert data["job_type"] == "test"
        assert data["status"] == "pending"
        assert data["metadata"] == {"key": "value"}
        assert "progress" in data
        assert data["progress"]["percentage"] == 0.0

    def test_job_to_dict_with_progress(self):
        """Test job serialization includes progress."""
        job = Job(id="test-1", job_type="test")
        job.progress.current = 25
        job.progress.total = 100
        job.progress.message = "Processing..."

        data = job.to_dict()
        assert data["progress"]["current"] == 25
        assert data["progress"]["total"] == 100
        assert data["progress"]["percentage"] == 25.0
        assert data["progress"]["message"] == "Processing..."


class TestJobManager:
    """Tests for JobManager."""

    @pytest.fixture
    def job_manager(self):
        """Create a fresh job manager for each test."""
        return JobManager(max_concurrent_jobs=2)

    @pytest.fixture
    def simple_worker(self):
        """Create a simple worker that completes immediately."""

        async def worker(job: Job, update_progress):
            update_progress(1, 1, "Done")
            return {"success": True}

        return worker

    @pytest.fixture
    def slow_worker(self):
        """Create a worker that takes time to complete."""

        async def worker(job: Job, update_progress):
            update_progress(0, 10, "Starting")
            for i in range(10):
                await asyncio.sleep(0.01)
                update_progress(i + 1, 10, f"Step {i + 1}")
            return {"steps": 10}

        return worker

    @pytest.fixture
    def failing_worker(self):
        """Create a worker that fails."""

        async def worker(job: Job, update_progress):
            update_progress(1, 2, "Working")
            raise ValueError("Something went wrong")

        return worker

    @pytest.mark.asyncio
    async def test_register_worker(self, job_manager, simple_worker):
        """Test registering a worker."""
        job_manager.register_worker("test", simple_worker)
        assert "test" in job_manager.registered_job_types

    @pytest.mark.asyncio
    async def test_submit_job_without_worker(self, job_manager):
        """Test submitting job without registered worker raises error."""
        with pytest.raises(ValueError, match="No worker registered"):
            await job_manager.submit_job("unknown_type")

    @pytest.mark.asyncio
    async def test_submit_job(self, job_manager, simple_worker):
        """Test submitting a job."""
        job_manager.register_worker("test", simple_worker)

        job_id = await job_manager.submit_job("test", metadata={"foo": "bar"})

        assert job_id is not None
        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.metadata == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_job_execution(self, job_manager, simple_worker):
        """Test job executes and completes."""
        job_manager.register_worker("test", simple_worker)

        job_id = await job_manager.submit_job("test")
        job = await job_manager.wait_for_job(job_id, timeout=5.0)

        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"success": True}
        assert job.started_at is not None
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_job_progress_tracking(self, job_manager, slow_worker):
        """Test progress is tracked during execution."""
        job_manager.register_worker("test", slow_worker)

        job_id = await job_manager.submit_job("test")

        # Wait a bit for job to start
        await asyncio.sleep(0.05)

        job = job_manager.get_job(job_id)
        assert job is not None
        # Job should be processing with some progress
        assert job.status == JobStatus.PROCESSING
        assert job.progress.total == 10

        # Wait for completion
        await job_manager.wait_for_job(job_id, timeout=5.0)
        assert job.status == JobStatus.COMPLETED
        assert job.progress.current == 10

    @pytest.mark.asyncio
    async def test_job_failure(self, job_manager, failing_worker):
        """Test job failure is recorded."""
        job_manager.register_worker("test", failing_worker)

        job_id = await job_manager.submit_job("test")
        job = await job_manager.wait_for_job(job_id, timeout=5.0)

        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"
        assert job.result is None

    @pytest.mark.asyncio
    async def test_get_job_status(self, job_manager, simple_worker):
        """Test getting job status."""
        job_manager.register_worker("test", simple_worker)

        job_id = await job_manager.submit_job("test")

        # Status exists
        status = job_manager.get_job_status(job_id)
        assert status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.COMPLETED)

        # Unknown job returns None
        assert job_manager.get_job_status("unknown") is None

    @pytest.mark.asyncio
    async def test_get_all_jobs(self, job_manager, simple_worker):
        """Test getting all jobs."""
        job_manager.register_worker("test", simple_worker)
        job_manager.register_worker("other", simple_worker)

        await job_manager.submit_job("test")
        await job_manager.submit_job("test")
        await job_manager.submit_job("other")

        all_jobs = job_manager.get_all_jobs()
        assert len(all_jobs) == 3

        test_jobs = job_manager.get_all_jobs(job_type="test")
        assert len(test_jobs) == 2

    @pytest.mark.asyncio
    async def test_cancel_job(self, job_manager):
        """Test cancelling a job."""

        async def long_worker(job: Job, update_progress):
            for i in range(100):
                await asyncio.sleep(0.1)
                update_progress(i, 100, "Working")
            return {"done": True}

        job_manager.register_worker("long", long_worker)

        job_id = await job_manager.submit_job("long")

        # Wait for job to start
        await asyncio.sleep(0.05)

        result = await job_manager.cancel_job(job_id)
        assert result is True

        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self, job_manager, simple_worker):
        """Test cancelling already completed job returns False."""
        job_manager.register_worker("test", simple_worker)

        job_id = await job_manager.submit_job("test")
        await job_manager.wait_for_job(job_id, timeout=5.0)

        result = await job_manager.cancel_job(job_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self, job_manager):
        """Test cancelling unknown job returns False."""
        result = await job_manager.cancel_job("unknown")
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_job_timeout(self, job_manager):
        """Test wait_for_job with timeout."""

        async def long_worker(job: Job, update_progress):
            await asyncio.sleep(10)  # Very long
            return {"done": True}

        job_manager.register_worker("long", long_worker)

        job_id = await job_manager.submit_job("long")

        # Wait with short timeout
        job = await job_manager.wait_for_job(job_id, timeout=0.1)

        # Should return the job even if not complete
        assert job is not None
        assert job.status in (JobStatus.PENDING, JobStatus.PROCESSING)

        # Cleanup
        await job_manager.cancel_job(job_id)

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, job_manager):
        """Test concurrent job limit is respected."""
        execution_order = []

        async def tracking_worker(job: Job, update_progress):
            execution_order.append(f"start_{job.metadata['num']}")
            await asyncio.sleep(0.1)
            execution_order.append(f"end_{job.metadata['num']}")
            return job.metadata["num"]

        job_manager.register_worker("track", tracking_worker)

        # Submit 4 jobs with limit of 2
        job_ids = []
        for i in range(4):
            job_id = await job_manager.submit_job("track", metadata={"num": i})
            job_ids.append(job_id)

        # Wait for all to complete
        for job_id in job_ids:
            await job_manager.wait_for_job(job_id, timeout=5.0)

        # Verify all completed
        for job_id in job_ids:
            job = job_manager.get_job(job_id)
            assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_active_job_count(self, job_manager):
        """Test active job count tracking."""

        async def slow_worker(job: Job, update_progress):
            await asyncio.sleep(0.5)
            return True

        job_manager.register_worker("slow", slow_worker)

        assert job_manager.active_job_count == 0

        job_id = await job_manager.submit_job("slow")

        # Give time for job to start
        await asyncio.sleep(0.05)
        assert job_manager.active_job_count >= 1

        await job_manager.wait_for_job(job_id, timeout=5.0)
        assert job_manager.active_job_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_completed_jobs(self, job_manager, simple_worker):
        """Test cleanup of old completed jobs."""
        job_manager.register_worker("test", simple_worker)

        job_id = await job_manager.submit_job("test")
        job = await job_manager.wait_for_job(job_id, timeout=5.0)

        # Manually set completed time to past
        job.completed_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Cleanup jobs older than 1 hour
        removed = job_manager.cleanup_completed_jobs(max_age_seconds=3600)
        assert removed == 1

        # Job should be gone
        assert job_manager.get_job(job_id) is None


class TestGlobalJobManager:
    """Tests for global job manager singleton."""

    def setup_method(self):
        """Reset global job manager before each test."""
        reset_job_manager()

    def teardown_method(self):
        """Reset global job manager after each test."""
        reset_job_manager()

    def test_get_job_manager(self):
        """Test getting global job manager."""
        manager1 = get_job_manager()
        manager2 = get_job_manager()

        assert manager1 is manager2

    def test_reset_job_manager(self):
        """Test resetting global job manager."""
        manager1 = get_job_manager()
        reset_job_manager()
        manager2 = get_job_manager()

        assert manager1 is not manager2
