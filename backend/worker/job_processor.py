"""Job processor - polls database for pending jobs and executes them."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from worker.config import WorkerConfig

logger = logging.getLogger(__name__)


class JobProcessor:
    """Processes ML jobs from the database queue.

    Polls for pending jobs, claims them atomically, executes the work,
    and updates job status. Designed to run as a separate process from
    the FastAPI backend.
    """

    def __init__(self, config: WorkerConfig) -> None:
        """Initialize the job processor.

        Args:
            config: Worker configuration.
        """
        self._config = config
        self._engine = None
        self._session_maker = None

    async def _init_db(self) -> None:
        """Initialize database connection."""
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            self._config.database_url,
            echo=False,
            future=True,
        )

        # Enable WAL mode for SQLite
        if self._config.database_url.startswith("sqlite"):

            @event.listens_for(self._engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Main processing loop.

        Args:
            shutdown_event: Event to signal graceful shutdown.
        """
        await self._init_db()
        logger.info(f"Worker {self._config.worker_id} starting job processing loop")

        consecutive_errors = 0

        while not shutdown_event.is_set():
            try:
                # Try to claim and process a job
                job_processed = await self._process_one_job()

                if job_processed:
                    consecutive_errors = 0
                else:
                    # No jobs available, wait before polling again
                    try:
                        await asyncio.wait_for(
                            shutdown_event.wait(),
                            timeout=self._config.poll_interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue polling

            except Exception as e:
                consecutive_errors += 1
                logger.exception(f"Error in job processing loop: {e}")

                if consecutive_errors >= self._config.max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({consecutive_errors}), stopping worker"
                    )
                    break

                # Back off on errors
                await asyncio.sleep(min(consecutive_errors * 2, 30))

        logger.info("Job processing loop ended")

    async def _process_one_job(self) -> bool:
        """Try to claim and process one job.

        Returns:
            True if a job was processed, False if no jobs available.
        """
        from app.models.processing_job import JobStatus, JobType, ProcessingJob

        async with self._session_maker() as session:
            # Find a pending job (oldest first)
            # Note: SQLite doesn't support skip_locked, so we use a simple select
            # and rely on the atomic update to handle concurrency
            stmt = (
                select(ProcessingJob)
                .where(ProcessingJob.status == JobStatus.PENDING)
                .order_by(ProcessingJob.created_at)
                .limit(1)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()

            if job is None:
                return False

            # Atomically claim the job - only succeeds if still PENDING
            # This handles race conditions with multiple workers
            claim_stmt = (
                update(ProcessingJob)
                .where(ProcessingJob.id == job.id)
                .where(ProcessingJob.status == JobStatus.PENDING)  # Only if still pending
                .values(
                    status=JobStatus.PROCESSING,
                    started_at=datetime.now(timezone.utc),
                    worker_id=self._config.worker_id,
                )
            )
            claim_result = await session.execute(claim_stmt)
            await session.commit()

            # Check if we actually claimed it (rowcount == 1)
            if claim_result.rowcount == 0:
                # Another worker got it first, try again
                logger.debug(f"Job {job.id} was claimed by another worker")
                return False

            # Refresh job data after claiming
            result = await session.execute(
                select(ProcessingJob).where(ProcessingJob.id == job.id)
            )
            job = result.scalar_one()

            logger.info(f"Claimed job {job.id} (type={job.job_type.value}, video_id={job.video_id})")

        # Process the job (outside the claiming transaction)
        try:
            if job.job_type == JobType.VIDEO_DETECTION:
                await self._process_video_detection(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

        except Exception as e:
            logger.exception(f"Job {job.id} failed: {e}")
            await self._mark_job_failed(job.id, str(e))
            return True

        return True

    async def _process_video_detection(self, job) -> None:
        """Process a video detection job.

        Args:
            job: The ProcessingJob to process.
        """
        from app.models.processing_job import JobStatus, ProcessingJob
        from app.models.video import Video

        video_id = job.video_id
        if video_id is None:
            raise ValueError("video_detection job requires video_id")

        params = job.parameters or {}
        sample_interval = params.get("sample_interval", 1)

        logger.info(f"Starting SAM3 detection for video {video_id}")

        # Get video path
        async with self._session_maker() as session:
            video_result = await session.execute(
                select(Video).where(Video.id == video_id)
            )
            video = video_result.scalar_one_or_none()

            if video is None:
                raise ValueError(f"Video {video_id} not found")

            video_path = Path(video.file_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Import SAM3 pipeline (heavy imports)
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        # Create progress callback
        async def update_progress(current: int, total: int, message: str = "") -> None:
            async with self._session_maker() as session:
                await session.execute(
                    update(ProcessingJob)
                    .where(ProcessingJob.id == job.id)
                    .values(
                        progress_current=current,
                        progress_total=total,
                        progress_message=message,
                    )
                )
                await session.commit()

        # Run detection
        pipeline = SAM3DetectionPipeline()
        frame_count = 0
        detection_count = 0

        async with self._session_maker() as session:
            from app.models.detection import PlayerDetection

            async for frame_detections in pipeline.process_video(
                video_path, sample_interval=sample_interval
            ):
                # Store detections
                for det in frame_detections.detections:
                    detection = PlayerDetection(
                        video_id=video_id,
                        frame_number=frame_detections.frame_number,
                        tracking_id=det.tracking_id,
                        bbox_x=det.bbox.x,
                        bbox_y=det.bbox.y,
                        bbox_width=det.bbox.width,
                        bbox_height=det.bbox.height,
                        confidence_score=det.confidence,
                    )
                    session.add(detection)
                    detection_count += 1

                frame_count += 1

                # Update progress every 10 frames
                if frame_count % 10 == 0:
                    await update_progress(frame_count, -1, f"Processed {frame_count} frames")

                # Checkpoint every 30 frames
                if frame_count % 30 == 0:
                    await session.commit()
                    logger.debug(f"Checkpointed at frame {frame_count}")

            await session.commit()

        # Mark job completed
        result = {
            "frames_processed": frame_count,
            "detections_stored": detection_count,
        }
        await self._mark_job_completed(job.id, result)

        logger.info(
            f"Job {job.id} completed: {frame_count} frames, {detection_count} detections"
        )

    async def _mark_job_completed(self, job_id: str, result: dict) -> None:
        """Mark a job as completed.

        Args:
            job_id: Job ID.
            result: Result dictionary.
        """
        from app.models.processing_job import JobStatus, ProcessingJob

        async with self._session_maker() as session:
            await session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == job_id)
                .values(
                    status=JobStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                    result=result,
                )
            )
            await session.commit()

    async def _mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job ID.
            error: Error message.
        """
        from app.models.processing_job import JobStatus, ProcessingJob

        async with self._session_maker() as session:
            await session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == job_id)
                .values(
                    status=JobStatus.FAILED,
                    completed_at=datetime.now(timezone.utc),
                    error_message=error,
                )
            )
            await session.commit()
