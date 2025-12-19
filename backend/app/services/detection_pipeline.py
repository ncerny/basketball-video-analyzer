"""Player detection pipeline service.

Integrates frame extraction, YOLO detection, and database storage
to process videos and identify players.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import settings
from app.ml.byte_tracker import PlayerTracker
from app.ml.types import Detection, FrameDetections
from app.ml.yolo_detector import YOLODetector
from app.models.detection import PlayerDetection
from app.models.video import ProcessingStatus, Video
from app.services.frame_extractor import FrameExtractor


@dataclass
class DetectionPipelineConfig:
    """Configuration for detection pipeline."""

    sample_interval: int = 3  # Extract every Nth frame
    batch_size: int = 8  # Frames per detection batch
    confidence_threshold: float = 0.5
    device: str = "cpu"
    delete_existing: bool = True  # Delete existing detections before processing
    enable_tracking: bool = True  # Enable ByteTrack player tracking
    tracking_buffer_seconds: float = 1.0  # How long to keep lost tracks
    tracking_iou_threshold: float = 0.8  # IOU threshold for matching


@dataclass
class DetectionPipelineResult:
    """Result from running detection pipeline."""

    video_id: int
    total_frames_processed: int
    total_detections: int
    persons_detected: int
    balls_detected: int
    error: str | None = None


# Progress callback type: (current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class DetectionPipeline:
    """Pipeline for detecting players in video frames.

    Orchestrates frame extraction, YOLO detection, and database storage.
    Designed to integrate with JobManager for background processing.
    """

    def __init__(
        self,
        db: AsyncSession,
        config: DetectionPipelineConfig | None = None,
    ) -> None:
        """Initialize the detection pipeline.

        Args:
            db: Async database session for storing results.
            config: Pipeline configuration (uses defaults if None).
        """
        self._db = db
        self._config = config or DetectionPipelineConfig(
            confidence_threshold=settings.yolo_confidence_threshold,
            device=self._resolve_device(settings.ml_device),
        )
        self._detector: YOLODetector | None = None
        self._tracker: PlayerTracker | None = None
        self._video_storage_path = Path(settings.video_storage_path)

    @staticmethod
    def _resolve_device(device_setting: str) -> str:
        """Resolve 'auto' device setting to actual device."""
        if device_setting == "auto":
            # Try to detect available device
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return device_setting

    def _get_detector(self) -> YOLODetector:
        """Get or create the YOLO detector instance."""
        if self._detector is None:
            self._detector = YOLODetector(
                model_path=settings.yolo_model_name,
                confidence_threshold=self._config.confidence_threshold,
                device=self._config.device,
            )
        return self._detector

    def _get_tracker(self, fps: float) -> PlayerTracker:
        """Get or create tracker instance.

        Args:
            fps: Video frame rate for motion prediction.

        Returns:
            PlayerTracker instance configured for this video.
        """
        if self._tracker is None:
            # Calculate buffer in frames
            buffer_frames = int(self._config.tracking_buffer_seconds * fps)

            self._tracker = PlayerTracker(
                track_activation_threshold=self._config.confidence_threshold,
                lost_track_buffer=buffer_frames,
                minimum_matching_threshold=self._config.tracking_iou_threshold,
                frame_rate=int(fps),
            )
        return self._tracker

    async def process_video(
        self,
        video_id: int,
        progress_callback: ProgressCallback | None = None,
    ) -> DetectionPipelineResult:
        """Process a video to detect players.

        Args:
            video_id: ID of video to process.
            progress_callback: Optional callback for progress updates.

        Returns:
            DetectionPipelineResult with processing statistics.
        """

        def report_progress(current: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(current, total, message)

        try:
            # Get video from database
            report_progress(0, 100, "Loading video metadata...")
            video = await self._get_video(video_id)

            if not video:
                return DetectionPipelineResult(
                    video_id=video_id,
                    total_frames_processed=0,
                    total_detections=0,
                    persons_detected=0,
                    balls_detected=0,
                    error=f"Video not found: {video_id}",
                )

            # Update status to processing
            await self._update_video_status(video, ProcessingStatus.PROCESSING)

            # Delete existing detections if configured
            if self._config.delete_existing:
                report_progress(5, 100, "Clearing existing detections...")
                await self._delete_existing_detections(video_id)

            # Get absolute video path
            video_path = self._video_storage_path / video.file_path

            if not video_path.exists():
                await self._update_video_status(video, ProcessingStatus.FAILED)
                return DetectionPipelineResult(
                    video_id=video_id,
                    total_frames_processed=0,
                    total_detections=0,
                    persons_detected=0,
                    balls_detected=0,
                    error=f"Video file not found: {video_path}",
                )

            # Run detection pipeline
            report_progress(10, 100, "Initializing detector...")
            result = await self._run_detection(video, video_path, report_progress)

            # Update video status
            if result.error:
                await self._update_video_status(video, ProcessingStatus.FAILED)
            else:
                await self._update_video_status(video, ProcessingStatus.COMPLETED, processed=True)

            return result

        except Exception as e:
            # Try to update status on any error
            try:
                video = await self._get_video(video_id)
                if video:
                    await self._update_video_status(video, ProcessingStatus.FAILED)
            except Exception:
                pass

            return DetectionPipelineResult(
                video_id=video_id,
                total_frames_processed=0,
                total_detections=0,
                persons_detected=0,
                balls_detected=0,
                error=str(e),
            )

    async def _get_video(self, video_id: int) -> Video | None:
        """Get video from database."""
        result = await self._db.execute(select(Video).where(Video.id == video_id))
        return result.scalar_one_or_none()

    async def _update_video_status(
        self,
        video: Video,
        status: ProcessingStatus,
        processed: bool | None = None,
    ) -> None:
        """Update video processing status."""
        video.processing_status = status
        if processed is not None:
            video.processed = processed
        await self._db.commit()

    async def _delete_existing_detections(self, video_id: int) -> None:
        """Delete existing detections for a video."""
        await self._db.execute(delete(PlayerDetection).where(PlayerDetection.video_id == video_id))
        await self._db.commit()

    async def _run_detection(
        self,
        video: Video,
        video_path: Path,
        report_progress: Callable[[int, int, str], None],
    ) -> DetectionPipelineResult:
        """Run the actual detection process."""
        detector = self._get_detector()
        total_detections = 0
        persons_detected = 0
        balls_detected = 0
        frames_processed = 0

        with FrameExtractor(video_path) as extractor:
            metadata = extractor.get_metadata()

            # Initialize tracker if enabled
            tracker = self._get_tracker(metadata.fps) if self._config.enable_tracking else None

            # Calculate total frames to process
            total_sampled_frames = extractor.count_sampled_frames(
                sample_interval=self._config.sample_interval
            )

            report_progress(15, 100, f"Processing {total_sampled_frames} frames...")

            # Process frames in batches
            batch_frames = []
            batch_frame_numbers = []

            for extracted_frame in extractor.extract_frames_sampled(
                sample_interval=self._config.sample_interval
            ):
                batch_frames.append(extracted_frame.frame)
                batch_frame_numbers.append(extracted_frame.frame_number)

                # Process batch when full
                if len(batch_frames) >= self._config.batch_size:
                    # Run CPU-intensive detection in thread to avoid blocking event loop
                    detections_list = await asyncio.to_thread(
                        detector.detect_batch,
                        batch_frames,
                        batch_frame_numbers[0]
                    )

                    # Apply tracking and store detections
                    for frame_detections in detections_list:
                        # Use frame_number from FrameDetections (already set by detect_batch)
                        # Apply tracking if enabled (also CPU-intensive, run in thread)
                        if tracker:
                            frame_detections = await asyncio.to_thread(
                                tracker.update,
                                frame_detections
                            )

                        stats = await self._store_frame_detections(
                            video.id, frame_detections.frame_number, frame_detections
                        )
                        total_detections += stats["total"]
                        persons_detected += stats["persons"]
                        balls_detected += stats["balls"]

                    frames_processed += len(batch_frames)
                    batch_frames = []
                    batch_frame_numbers = []

                    # Update progress (15% to 95% for detection)
                    progress = 15 + int((frames_processed / total_sampled_frames) * 80)
                    report_progress(
                        progress,
                        100,
                        f"Processed {frames_processed}/{total_sampled_frames} frames",
                    )

            # Process remaining frames
            if batch_frames:
                # Run CPU-intensive detection in thread to avoid blocking event loop
                detections_list = await asyncio.to_thread(
                    detector.detect_batch,
                    batch_frames,
                    batch_frame_numbers[0]
                )

                for frame_detections in detections_list:
                    # Use frame_number from FrameDetections (already set by detect_batch)
                    # Apply tracking if enabled (also CPU-intensive, run in thread)
                    if tracker:
                        frame_detections = await asyncio.to_thread(
                            tracker.update,
                            frame_detections
                        )

                    stats = await self._store_frame_detections(
                        video.id, frame_detections.frame_number, frame_detections
                    )
                    total_detections += stats["total"]
                    persons_detected += stats["persons"]
                    balls_detected += stats["balls"]

                frames_processed += len(batch_frames)

            report_progress(95, 100, "Finalizing...")

        # Commit all detections
        await self._db.commit()
        report_progress(100, 100, "Complete")

        return DetectionPipelineResult(
            video_id=video.id,
            total_frames_processed=frames_processed,
            total_detections=total_detections,
            persons_detected=persons_detected,
            balls_detected=balls_detected,
        )

    async def _store_frame_detections(
        self,
        video_id: int,
        frame_number: int,
        frame_detections: FrameDetections,
    ) -> dict[str, int]:
        """Store detections for a single frame.

        Note: tracking_id now comes from ByteTrack and is persistent across frames
        when tracking is enabled. Falls back to per-frame index if tracking disabled.

        Returns:
            Dict with counts: {"total", "persons", "balls"}
        """
        stats = {"total": 0, "persons": 0, "balls": 0}

        for i, detection in enumerate(frame_detections.detections):
            # Create PlayerDetection record
            player_detection = PlayerDetection(
                video_id=video_id,
                frame_number=frame_number,
                player_id=None,  # Assigned later through UI/OCR
                bbox_x=detection.bbox.x,
                bbox_y=detection.bbox.y,
                bbox_width=detection.bbox.width,
                bbox_height=detection.bbox.height,
                tracking_id=detection.tracking_id if detection.tracking_id is not None else i,
                confidence_score=detection.confidence,
            )
            self._db.add(player_detection)

            stats["total"] += 1
            if detection.is_person:
                stats["persons"] += 1
            else:
                stats["balls"] += 1

        return stats


async def create_detection_job_worker(job_manager):
    """Create and register a detection job worker with the job manager.

    This sets up the pipeline to work with the background job system.
    """
    from app.database import async_session_maker
    from app.services.job_manager import Job

    async def detection_worker(job: Job, update_progress: Callable[[int, int, str], None]):
        """Worker function for processing video detection jobs."""
        video_id = job.metadata.get("video_id")
        if not video_id:
            raise ValueError("video_id required in job metadata")

        async with async_session_maker() as db:
            config = DetectionPipelineConfig(
                sample_interval=job.metadata.get("sample_interval", 3),
                batch_size=job.metadata.get("batch_size", 8),
                confidence_threshold=job.metadata.get(
                    "confidence_threshold", settings.yolo_confidence_threshold
                ),
            )
            pipeline = DetectionPipeline(db, config)
            result = await pipeline.process_video(video_id, update_progress)

            if result.error:
                raise RuntimeError(result.error)

            return {
                "video_id": result.video_id,
                "total_frames_processed": result.total_frames_processed,
                "total_detections": result.total_detections,
                "persons_detected": result.persons_detected,
                "balls_detected": result.balls_detected,
            }

    job_manager.register_worker("video_detection", detection_worker)
