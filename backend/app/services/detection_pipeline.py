"""Player detection pipeline service.

Integrates frame extraction, YOLO detection, and database storage
to process videos and identify players.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import settings
from app.ml.byte_tracker import PlayerTracker
from app.ml.court_detector import CourtDetector
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
    tracking_buffer_seconds: float = 5.0  # How long to keep lost tracks (increased for occlusions)
    tracking_iou_threshold: float = (
        0.2  # IOU threshold for matching (lowered for fast basketball motion)
    )
    enable_court_detection: bool = True  # Enable court boundary detection
    court_overlap_threshold: float = 0.2  # Minimum overlap with court to keep detection


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

# Logger for performance metrics
logger = logging.getLogger(__name__)


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
        resolved_device = self._resolve_device(settings.ml_device)
        optimal_batch_size = self._get_optimal_batch_size(resolved_device)
        self._config = config or DetectionPipelineConfig(
            confidence_threshold=settings.yolo_confidence_threshold,
            device=resolved_device,
            batch_size=optimal_batch_size,
        )

        # Log device and batch size configuration
        logger.info(
            f"Detection pipeline initialized: device={resolved_device}, "
            f"batch_size={optimal_batch_size}, "
            f"confidence_threshold={self._config.confidence_threshold}"
        )

        self._detector: YOLODetector | None = None
        self._tracker: PlayerTracker | None = None
        self._court_detector: CourtDetector | None = None
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

    @staticmethod
    def _get_optimal_batch_size(device: str) -> int:
        """Get optimal batch size based on device type."""
        if device == "cuda":
            return settings.yolo_batch_size_cuda
        elif device == "mps":
            return settings.yolo_batch_size_mps
        else:
            return settings.yolo_batch_size_cpu

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

    def _get_court_detector(self) -> CourtDetector:
        """Get or create court detector instance.

        Returns:
            CourtDetector instance for detecting court boundaries.
        """
        if self._court_detector is None:
            self._court_detector = CourtDetector()
        return self._court_detector

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
            # Bug fix: Use effective FPS based on sample interval, not raw video FPS
            # The tracker only sees sampled frames, so motion prediction must account for this
            if self._config.enable_tracking:
                effective_fps = metadata.fps / self._config.sample_interval
                tracker = self._get_tracker(effective_fps)
                tracker.reset()  # Ensure clean state for new video
            else:
                tracker = None

            # Initialize court detector if enabled
            court_detector = (
                self._get_court_detector() if self._config.enable_court_detection else None
            )

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
                    batch_start_time = time.time()
                    detections_list = await asyncio.to_thread(
                        detector.detect_batch, batch_frames, batch_frame_numbers[0]
                    )
                    batch_inference_time = time.time() - batch_start_time

                    # Log performance metrics if enabled
                    if settings.enable_inference_timing:
                        fps = len(batch_frames) / batch_inference_time
                        logger.info(
                            f"Detection batch: {len(batch_frames)} frames in "
                            f"{batch_inference_time:.2f}s ({fps:.1f} fps) on {self._config.device}"
                        )

                    # Apply tracking and court detection
                    # Bug fix: Track BEFORE court filtering to maintain stable track IDs
                    # Filter persons only for tracking (balls cause spurious associations)
                    for i, frame_detections in enumerate(detections_list):
                        frame = batch_frames[i]

                        # 1. Separate persons from balls (track persons only)
                        person_detections, other_detections = self._filter_persons_only(
                            frame_detections
                        )

                        # 2. Apply tracking to persons FIRST (before court filtering)
                        if tracker:
                            person_detections = await asyncio.to_thread(
                                tracker.update, person_detections
                            )

                        # 3. Apply court filter to tracked persons
                        if court_detector:
                            person_detections = await asyncio.to_thread(
                                self._filter_detections_by_court,
                                person_detections,
                                frame,
                                court_detector,
                            )

                        # 4. Combine filtered persons with other detections (balls)
                        final_detections = FrameDetections(
                            frame_number=frame_detections.frame_number,
                            detections=person_detections.detections + other_detections,
                            frame_width=frame_detections.frame_width,
                            frame_height=frame_detections.frame_height,
                        )

                        stats = await self._store_frame_detections(
                            video.id, final_detections.frame_number, final_detections
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
                batch_start_time = time.time()
                detections_list = await asyncio.to_thread(
                    detector.detect_batch, batch_frames, batch_frame_numbers[0]
                )
                batch_inference_time = time.time() - batch_start_time

                # Log performance metrics if enabled
                if settings.enable_inference_timing:
                    fps = len(batch_frames) / batch_inference_time
                    logger.info(
                        f"Detection final batch: {len(batch_frames)} frames in "
                        f"{batch_inference_time:.2f}s ({fps:.1f} fps) on {self._config.device}"
                    )

                # Apply tracking and court detection (same as main batch)
                for i, frame_detections in enumerate(detections_list):
                    frame = batch_frames[i]

                    # 1. Separate persons from balls (track persons only)
                    person_detections, other_detections = self._filter_persons_only(
                        frame_detections
                    )

                    # 2. Apply tracking to persons FIRST (before court filtering)
                    if tracker:
                        person_detections = await asyncio.to_thread(
                            tracker.update, person_detections
                        )

                    # 3. Apply court filter to tracked persons
                    if court_detector:
                        person_detections = await asyncio.to_thread(
                            self._filter_detections_by_court,
                            person_detections,
                            frame,
                            court_detector,
                        )

                    # 4. Combine filtered persons with other detections (balls)
                    final_detections = FrameDetections(
                        frame_number=frame_detections.frame_number,
                        detections=person_detections.detections + other_detections,
                        frame_width=frame_detections.frame_width,
                        frame_height=frame_detections.frame_height,
                    )

                    stats = await self._store_frame_detections(
                        video.id, final_detections.frame_number, final_detections
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

    def _filter_persons_only(
        self, frame_detections: FrameDetections
    ) -> tuple[FrameDetections, list[Detection]]:
        """Separate person and non-person detections.

        Ball detections (class_id=32) can cause spurious associations in the tracker.
        We track persons only and keep balls separate.

        Args:
            frame_detections: All detections from YOLO.

        Returns:
            Tuple of (persons_only FrameDetections, list of other detections like balls)
        """
        persons = [d for d in frame_detections.detections if d.is_person]
        others = [d for d in frame_detections.detections if not d.is_person]
        return (
            FrameDetections(
                frame_number=frame_detections.frame_number,
                detections=persons,
                frame_width=frame_detections.frame_width,
                frame_height=frame_detections.frame_height,
            ),
            others,
        )

    def _filter_detections_by_court(
        self,
        frame_detections: FrameDetections,
        frame: Any,
        court_detector: CourtDetector,
    ) -> FrameDetections:
        """Filter detections to only include those within court boundaries.

        Args:
            frame_detections: All detections from YOLO.
            frame: The video frame (numpy array).
            court_detector: Court detector instance.

        Returns:
            Filtered FrameDetections with only in-court detections.
        """
        # Detect court mask for this frame
        court_mask = court_detector.detect_court_mask(frame)

        # Filter detections
        filtered_detections = []
        for detection in frame_detections.detections:
            # Check if bounding box overlaps with court
            is_in_court = court_detector.is_bbox_in_court(
                bbox_x=detection.bbox.x,
                bbox_y=detection.bbox.y,
                bbox_width=detection.bbox.width,
                bbox_height=detection.bbox.height,
                mask=court_mask,
                threshold=self._config.court_overlap_threshold,
            )

            if is_in_court:
                filtered_detections.append(detection)

        # Return new FrameDetections with filtered list
        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=filtered_detections,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
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
