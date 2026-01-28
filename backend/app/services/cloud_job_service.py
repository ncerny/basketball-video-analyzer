"""Cloud job submission service.

Manages cloud GPU processing job submission:
- Creates job manifests in R2
- Auto-starts RunPod workers
"""

import logging
import uuid
from datetime import datetime, timezone

from app.config import settings

logger = logging.getLogger(__name__)


class CloudJobService:
    """Service for submitting cloud GPU processing jobs."""

    def __init__(self) -> None:
        """Initialize cloud job service."""
        self._storage = None
        self._runpod = None

    def _ensure_storage(self):
        """Lazy initialize CloudStorage."""
        if self._storage is None:
            from worker.cloud_storage import CloudStorage

            if not settings.r2_account_id:
                raise RuntimeError("R2 storage is not configured")

            self._storage = CloudStorage(
                account_id=settings.r2_account_id,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_bucket_name,
            )
        return self._storage

    def _ensure_runpod(self):
        """Lazy initialize RunPod service."""
        if self._runpod is None:
            from worker.runpod_service import get_runpod_service

            self._runpod = get_runpod_service()
        return self._runpod

    @property
    def is_configured(self) -> bool:
        """Check if cloud job service is properly configured."""
        return bool(settings.r2_account_id)

    def submit_job(
        self,
        video_id: int,
        r2_key: str,
        sample_interval: int = 1,
        confidence_threshold: float = 0.25,
        auto_start_worker: bool = True,
    ) -> str:
        """Submit a video for cloud GPU processing.

        Creates a job manifest in R2 that will be picked up by the cloud worker.
        Optionally auto-starts the RunPod worker if not already running.

        Args:
            video_id: Database video ID.
            r2_key: R2 storage key for the video (permanent location).
            sample_interval: Process every Nth frame (default 1 = all frames).
            confidence_threshold: Detection confidence threshold.
            auto_start_worker: Whether to auto-start RunPod if not running.

        Returns:
            Job ID (UUID string).

        Raises:
            RuntimeError: If R2 storage is not configured.
        """
        storage = self._ensure_storage()

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job manifest
        from worker.cloud_storage import JobManifest

        manifest = JobManifest(
            job_id=job_id,
            video_id=video_id,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            parameters={
                "sample_interval": sample_interval,
                "confidence_threshold": confidence_threshold,
                "video_key": r2_key,  # Points to permanent video location
            },
        )

        # Upload manifest to R2
        storage.upload_job_manifest(manifest)
        logger.info(f"Job submitted: {job_id} for video {video_id}")

        # Auto-start RunPod worker if configured
        if auto_start_worker:
            self._try_start_worker()

        return job_id

    def _try_start_worker(self) -> bool:
        """Try to start RunPod worker if not running.

        Returns:
            True if worker is running or was started, False otherwise.
        """
        try:
            runpod = self._ensure_runpod()

            # Check if RunPod is configured
            if not runpod._api_key:
                logger.debug("RunPod not configured - worker will not auto-start")
                return False

            if runpod.is_pod_running():
                logger.debug("RunPod worker already running")
                return True

            logger.info("Starting RunPod worker...")
            if runpod.start_pod():
                logger.info("RunPod worker start requested")
                return True
            else:
                logger.warning("Failed to start RunPod worker")
                return False

        except Exception as e:
            logger.warning(f"Failed to check/start RunPod worker: {e}")
            return False

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a submitted job.

        Args:
            job_id: Job ID to check.

        Returns:
            Job manifest as dict, or None if not found.
        """
        try:
            storage = self._ensure_storage()
            manifest = storage.get_job_manifest(job_id)
            if manifest:
                return manifest.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None


# Singleton instance
_cloud_job_service: CloudJobService | None = None


def get_cloud_job_service() -> CloudJobService:
    """Get the cloud job service singleton.

    Returns:
        CloudJobService instance.
    """
    global _cloud_job_service
    if _cloud_job_service is None:
        _cloud_job_service = CloudJobService()
    return _cloud_job_service
