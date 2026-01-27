# backend/worker/cloud_worker.py
"""Cloud worker that polls R2 for jobs and processes them."""

import asyncio
import gc
import logging
import signal
import socket
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from worker.cloud_storage import CloudStorage, JobManifest
from worker.config import WorkerConfig
from worker.runpod_service import get_runpod_service

logger = logging.getLogger(__name__)


class CloudWorker:
    """Worker that polls R2 for pending jobs and processes them."""

    def __init__(self, storage: CloudStorage, config: WorkerConfig) -> None:
        self._storage = storage
        self._config = config
        self._worker_id = config.worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._last_job_time: float = time.time()  # Track when we last processed a job
        self._cache_uploaded: bool = False  # Track if we've uploaded torch cache this session

    async def _warmup_model(self) -> None:
        """Warmup model by downloading cache or compiling before accepting jobs.

        This ensures the first real job doesn't pay the compilation cost.
        """
        import numpy as np
        from PIL import Image

        # Step 1: Try to download existing torch compile cache from R2
        logger.info("Checking for existing torch compile cache in R2...")
        cache_exists = False
        try:
            cache_exists = await asyncio.to_thread(self._storage.download_torch_cache)
            if cache_exists:
                logger.info("Restored torch compile cache from R2")
        except Exception as e:
            logger.warning(f"Failed to restore torch cache: {e}")

        # Step 2: If no cache, run warmup inference to compile the model
        if not cache_exists and torch.cuda.is_available():
            logger.info("No cache found - warming up model (this triggers compilation)...")
            try:
                from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

                # Create tracker with torch.compile enabled
                # Use same prompt as production to ensure identical code paths
                config = SAM3TrackerConfig(
                    prompt="basketball player",
                    use_torch_compile=True,
                    use_half_precision=True,
                )
                tracker = SAM3VideoTracker(config)

                # Run a dummy inference to trigger compilation
                # Create a small fake "video" - just a few frames
                logger.info("Running warmup inference to compile model graphs...")
                dummy_frames = [
                    Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                    for _ in range(3)
                ]

                # Process the dummy frames (this triggers torch.compile)
                for frame_idx, frame in enumerate(dummy_frames):
                    try:
                        # Call the model's internal method to process a frame
                        tracker._load_predictor()  # Ensure model is loaded
                        # Process through the model to trigger compilation
                        inputs = tracker._processor(
                            images=frame,
                            return_tensors="pt"
                        ).to(tracker._device)
                        with torch.no_grad():
                            _ = tracker._model.image_encoder(inputs["pixel_values"])
                        logger.info(f"Warmup frame {frame_idx + 1}/3 processed")
                    except Exception as e:
                        logger.warning(f"Warmup frame {frame_idx + 1} failed: {e}")
                        break

                # Clean up
                del tracker
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("Model warmup complete")

                # Step 3: Upload the compiled cache to R2 for future pods
                logger.info("Uploading torch compile cache to R2...")
                try:
                    uploaded = await asyncio.to_thread(self._storage.upload_torch_cache)
                    if uploaded:
                        self._cache_uploaded = True
                        logger.info("Torch compile cache uploaded to R2")
                except Exception as e:
                    logger.warning(f"Failed to upload torch cache: {e}")

            except Exception as e:
                logger.error(f"Model warmup failed: {e}")
                # Continue anyway - first job will just be slower
        else:
            if cache_exists:
                logger.info("Using cached compilation from R2")
            elif not torch.cuda.is_available():
                logger.info("No CUDA available, skipping warmup")

    async def run(self, shutdown_event: asyncio.Event, single_job: bool = False) -> None:
        """Main processing loop.

        Args:
            shutdown_event: Event to signal graceful shutdown.
            single_job: If True, exit after processing one job.
        """
        logger.info(f"Cloud worker {self._worker_id} starting...")

        # Warmup: Download cache or compile model BEFORE accepting jobs
        await self._warmup_model()

        # Reset any orphaned jobs from previous crashed workers
        try:
            reset_jobs = await asyncio.to_thread(self._storage.reset_orphaned_jobs)
            if reset_jobs:
                logger.info(f"Reset {len(reset_jobs)} orphaned jobs: {reset_jobs}")
        except Exception as e:
            logger.warning(f"Failed to reset orphaned jobs: {e}")

        while not shutdown_event.is_set():
            # Poll for pending jobs (wrap blocking I/O in thread for proper async)
            try:
                jobs = await asyncio.to_thread(self._storage.list_pending_jobs)
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

                # Log idle time (API handles idle shutdown to avoid needing RUNPOD_API_KEY here)
                idle_seconds = time.time() - self._last_job_time
                logger.debug(f"No pending jobs (idle {idle_seconds:.0f}s), waiting...")
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
                # Reset idle timer after successful job
                self._last_job_time = time.time()
            except Exception as e:
                logger.exception(f"Job {job.job_id} failed: {e}")
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc).isoformat()
                try:
                    await asyncio.to_thread(self._storage.upload_job_manifest, job)
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
        await asyncio.to_thread(self._storage.upload_job_manifest, job)

        # Download video to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get video key from manifest (includes correct extension)
            video_key = job.parameters.get("video_key", f"videos/{job.job_id}.mp4")
            video_suffix = Path(video_key).suffix or ".mp4"
            video_path = Path(tmpdir) / f"{job.job_id}{video_suffix}"
            try:
                logger.info(f"Downloading video for job {job.job_id} from {video_key}...")
                await asyncio.to_thread(
                    self._storage.download_video_by_key, video_key, video_path
                )
            except Exception as e:
                logger.error(f"Video download failed for job {job.job_id}: {e}")
                raise  # Re-raise; tempdir cleanup handled by context manager

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
            await asyncio.to_thread(self._storage.upload_results, job.job_id, results)

        # Mark as completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(self._storage.upload_job_manifest, job)

        logger.info(f"Job {job.job_id} completed: {len(detections)} detections")

        # Upload torch compile cache after first job (for future pods)
        if not self._cache_uploaded:
            try:
                uploaded = await asyncio.to_thread(self._storage.upload_torch_cache)
                if uploaded:
                    self._cache_uploaded = True
            except Exception as e:
                logger.warning(f"Failed to upload torch cache: {e}")

    async def _run_detection(self, job: JobManifest, video_path: Path) -> list[dict]:
        """Run SAM3 detection on video.

        Args:
            job: Job manifest with parameters.
            video_path: Path to downloaded video.

        Returns:
            List of detection dicts.
        """
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor
        from app.ml.sam3_detection_pipeline import SAM3DetectionPipeline

        params = job.parameters
        sample_interval = params.get("sample_interval", 1)
        confidence = params.get("confidence_threshold", 0.25)

        # Get total frame count for progress reporting
        extractor = SAM3FrameExtractor()
        total_frames = extractor.get_video_frame_count(video_path, sample_interval)
        logger.info(f"Starting SAM3 detection (sample_interval={sample_interval}, total_frames={total_frames})")

        # Initialize pipeline inside try block to ensure job state is correct on failure
        pipeline = None
        detections = []
        frame_count = 0

        try:
            pipeline = SAM3DetectionPipeline(confidence_threshold=confidence)

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

                # Update progress every 50 frames (async to allow signal handling)
                if frame_count % 50 == 0:
                    try:
                        await asyncio.to_thread(
                            self._storage.update_status,
                            job.job_id,
                            frame_count,
                            total_frames,
                            f"Processing frame {frame_count}/{total_frames}",
                        )
                        # Update heartbeat and manifest
                        job.last_heartbeat = datetime.now(timezone.utc).isoformat()
                        await asyncio.to_thread(
                            self._storage.upload_job_manifest, job
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update progress for {job.job_id}: {e}")

        finally:
            # Release GPU/MPS resources held by the pipeline
            if pipeline is not None:
                del pipeline
            gc.collect()

            # Clear MPS cache if available (Apple Silicon GPU memory)
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception as e:
                    logger.debug(f"MPS cache clear failed (non-critical): {e}")

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"CUDA cache clear failed (non-critical): {e}")

            logger.debug("Pipeline resources released")

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
