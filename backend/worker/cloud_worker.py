# backend/worker/cloud_worker.py
"""Cloud worker that polls R2 for jobs and processes them."""

import asyncio
import logging
import signal
import socket
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from worker.cloud_storage import CloudStorage, JobManifest
from worker.config import WorkerConfig

logger = logging.getLogger(__name__)


class CloudWorker:
    """Worker that polls R2 for pending jobs and processes them."""

    def __init__(self, storage: CloudStorage, config: WorkerConfig) -> None:
        self._storage = storage
        self._config = config
        self._worker_id = config.worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    async def run(self, shutdown_event: asyncio.Event, single_job: bool = False) -> None:
        """Main processing loop.

        Args:
            shutdown_event: Event to signal graceful shutdown.
            single_job: If True, exit after processing one job.
        """
        logger.info(f"Cloud worker {self._worker_id} starting...")

        while not shutdown_event.is_set():
            # Poll for pending jobs
            try:
                jobs = self._storage.list_pending_jobs()
            except Exception as e:
                logger.error(f"Failed to list pending jobs: {e}")
                if single_job:
                    logger.info("Error listing jobs in single_job mode, exiting")
                    break
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=self._config.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            if not jobs:
                if single_job:
                    logger.info("No pending jobs found, exiting")
                    break
                logger.debug("No pending jobs, waiting...")
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=self._config.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            # Process first pending job
            job = jobs[0]
            logger.info(f"Claiming job {job.job_id} (video_id={job.video_id})")

            try:
                await self._process_job(job)
            except Exception as e:
                logger.exception(f"Job {job.job_id} failed: {e}")
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc).isoformat()
                try:
                    self._storage.upload_job_manifest(job)
                except Exception as upload_err:
                    logger.error(f"Failed to upload failure status for {job.job_id}: {upload_err}")

            if single_job:
                break

        logger.info("Cloud worker stopped")

    async def _process_job(self, job: JobManifest) -> None:
        """Process a single job."""
        # Mark as processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.worker_id = self._worker_id
        self._storage.upload_job_manifest(job)

        # Download video to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / f"{job.job_id}.mp4"
            self._storage.download_video(job.job_id, video_path)

            # Run detection
            detections = await self._run_detection(job, video_path)

            # Upload results
            results = {
                "job_id": job.job_id,
                "video_id": job.video_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "frames_processed": job.frames_processed,
                "detections": detections,
            }
            self._storage.upload_results(job.job_id, results)

        # Mark as completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._storage.upload_job_manifest(job)

        logger.info(f"Job {job.job_id} completed: {len(detections)} detections")

    async def _run_detection(self, job: JobManifest, video_path: Path) -> list[dict]:
        """Run SAM3 detection on video.

        Args:
            job: Job manifest with parameters.
            video_path: Path to downloaded video.

        Returns:
            List of detection dicts.
        """
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        params = job.parameters
        sample_interval = params.get("sample_interval", 1)
        confidence = params.get("confidence_threshold", 0.25)

        logger.info(f"Starting SAM3 detection (sample_interval={sample_interval})")

        pipeline = SAM3DetectionPipeline(confidence_threshold=confidence)
        detections = []
        frame_count = 0

        async for frame_detections in pipeline.process_video(
            video_path, sample_interval=sample_interval
        ):
            for det in frame_detections.detections:
                detections.append({
                    "frame": frame_detections.frame_number,
                    "track_id": det.tracking_id,
                    "bbox": [det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height],
                    "confidence": det.confidence,
                })

            frame_count += 1
            job.frames_processed = frame_count

            # Update progress every 50 frames
            if frame_count % 50 == 0:
                try:
                    self._storage.update_status(
                        job.job_id,
                        current=frame_count,
                        total=-1,  # Unknown total
                        message=f"Processing frame {frame_count}",
                    )
                    # Also update manifest
                    self._storage.upload_job_manifest(job)
                except Exception as e:
                    logger.warning(f"Failed to update progress for {job.job_id}: {e}")

        return detections


async def main():
    """Entry point for cloud worker."""
    import os
    from app.config import settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not settings.r2_account_id:
        logger.error("R2 not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    storage = CloudStorage(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
    )

    config = WorkerConfig(
        poll_interval_seconds=settings.cloud_worker_poll_interval,
        worker_id=os.environ.get("WORKER_ID", ""),
    )

    worker = CloudWorker(storage, config)

    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for single-job mode
    single_job = os.environ.get("SINGLE_JOB", "").lower() == "true"

    await worker.run(shutdown_event, single_job=single_job)


if __name__ == "__main__":
    asyncio.run(main())
