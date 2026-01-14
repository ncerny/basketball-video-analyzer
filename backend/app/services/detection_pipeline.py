"""Player detection pipeline service.

Integrates frame extraction, YOLO detection, and database storage
to process videos and identify players.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import settings
from app.ml.base import BaseDetector
from app.ml.byte_tracker import PlayerTracker
from app.ml.color_extractor import extract_jersey_color
from app.ml.court_detector import CourtDetector
from app.ml.jersey_ocr import JerseyOCR, OCRConfig, OCRResult
from app.ml.legibility_filter import LegibilityConfig, check_legibility, extract_jersey_crop
from app.ml.norfair_tracker import NorfairTracker
from app.ml.sam2_tracker import SAM2TrackerConfig, SAM2VideoTracker
from app.ml.types import BoundingBox, Detection, FrameDetections
from app.ml.yolo_detector import YOLODetector
from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber
from app.models.video import ProcessingStatus, Video
from app.services.batch_orchestrator import OrchestratorConfig, SequentialOrchestrator
from app.services.frame_extractor import FrameExtractor
from app.services.track_merger import TrackMerger, TrackMergerConfig


@dataclass
class PipelinePhase:
    start_pct: int
    end_pct: int
    number: int
    name: str

    def progress(self, phase_pct: float, detail: str, total_phases: int) -> tuple[int, str]:
        phase_range = self.end_pct - self.start_pct
        overall_pct = int(self.start_pct + (phase_pct * phase_range))
        message = f"[{self.number}/{total_phases}] {self.name}: {detail}"
        return overall_pct, message


PHASE_SETUP = PipelinePhase(start_pct=0, end_pct=10, number=1, name="Setup")
PHASE_DETECTION = PipelinePhase(start_pct=10, end_pct=85, number=2, name="Detection")
PHASE_FINALIZE = PipelinePhase(start_pct=85, end_pct=92, number=3, name="Finalize")
PHASE_TRACK_MERGE = PipelinePhase(start_pct=92, end_pct=100, number=4, name="Track Merge")
TOTAL_PHASES = 4


@dataclass
class DetectionPipelineConfig:
    sample_interval: int = 3
    batch_size: int = 8
    confidence_threshold: float = 0.5
    device: str = "cpu"
    delete_existing: bool = True
    enable_tracking: bool = True
    track_activation_threshold: float = 0.25
    tracking_buffer_seconds: float = 5.0
    tracking_iou_threshold: float = 0.35
    enable_court_detection: bool = True
    court_overlap_threshold: float = 0.2
    max_seconds: float | None = None
    enable_track_merging: bool = True
    merge_max_temporal_gap_frames: int = 30
    merge_max_spatial_distance: float = 300.0
    merge_min_size_similarity: float = 0.6
    enable_jersey_ocr: bool = True
    ocr_sample_rate: int = 10


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
            enable_jersey_ocr=settings.enable_jersey_ocr,
        )

        # Log device and batch size configuration
        logger.info(
            f"Detection pipeline initialized: device={resolved_device}, "
            f"batch_size={optimal_batch_size}, "
            f"confidence_threshold={self._config.confidence_threshold}"
        )

        self._detector: BaseDetector | None = None
        self._tracker: PlayerTracker | NorfairTracker | SAM2VideoTracker | None = None
        self._court_detector: CourtDetector | None = None
        self._jersey_ocr: JerseyOCR | None = None
        self._legibility_config = LegibilityConfig()
        self._ocr_frame_counts: dict[int, int] = {}
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

    def _get_detector(self) -> BaseDetector:
        if self._detector is None:
            if settings.detection_backend == "rfdetr":
                from app.ml.rfdetr_detector import RFDETRDetector

                logger.info("Using RF-DETR detection backend")
                self._detector = RFDETRDetector(
                    confidence_threshold=self._config.confidence_threshold,
                    device=self._config.device,
                )
            else:
                logger.info("Using YOLO detection backend")
                self._detector = YOLODetector(
                    model_path=settings.yolo_model_name,
                    confidence_threshold=self._config.confidence_threshold,
                    device=self._config.device,
                )
        return self._detector

    def _get_tracker(self, fps: float) -> PlayerTracker | NorfairTracker | SAM2VideoTracker:
        if self._tracker is None:
            buffer_frames = int(self._config.tracking_buffer_seconds * fps)

            if settings.tracking_backend == "sam2":
                logger.info("Using SAM2 tracking backend (mask-based)")
                config = SAM2TrackerConfig(
                    model_name=settings.sam2_model_name,
                    device=self._config.device,
                    min_iou_threshold=settings.sam2_new_object_iou_threshold,
                    lost_track_frames=buffer_frames,
                )
                self._tracker = SAM2VideoTracker(config)
            elif settings.tracking_backend == "norfair":
                logger.info("Using Norfair tracking backend (Euclidean distance)")
                self._tracker = NorfairTracker(
                    distance_threshold=250.0,
                    hit_counter_max=buffer_frames,
                    initialization_delay=1,
                )
            else:
                logger.info("Using ByteTrack tracking backend (IOU)")
                self._tracker = PlayerTracker(
                    track_activation_threshold=self._config.track_activation_threshold,
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

    def _get_jersey_ocr(self) -> JerseyOCR:
        if self._jersey_ocr is None:
            config = OCRConfig(
                model_name=settings.ocr_model_name,
                device=self._config.device,
            )
            self._jersey_ocr = JerseyOCR(config)
        return self._jersey_ocr

    def _should_run_ocr_for_track(self, tracking_id: int) -> bool:
        count = self._ocr_frame_counts.get(tracking_id, 0)
        self._ocr_frame_counts[tracking_id] = count + 1
        return count % self._config.ocr_sample_rate == 0

    def _reset_ocr_state(self) -> None:
        self._ocr_frame_counts.clear()

    async def _run_ocr_on_frame(
        self,
        video_id: int,
        frame_number: int,
        frame: np.ndarray,
        detections: FrameDetections,
    ) -> int:
        if not self._config.enable_jersey_ocr:
            return 0

        ocr = self._get_jersey_ocr()
        ocr_count = 0

        for detection in detections.detections:
            if not detection.is_person:
                continue

            if detection.tracking_id is None:
                continue

            if not self._should_run_ocr_for_track(detection.tracking_id):
                continue

            bbox = BoundingBox(
                x=detection.bbox.x,
                y=detection.bbox.y,
                width=detection.bbox.width,
                height=detection.bbox.height,
            )

            legibility = check_legibility(
                frame, bbox, detection.confidence, self._legibility_config
            )
            if not legibility.is_legible:
                continue

            crop = extract_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue

            ocr_result: OCRResult = await asyncio.to_thread(ocr.read_jersey_number, crop)

            jersey_number = JerseyNumber(
                video_id=video_id,
                frame_number=frame_number,
                tracking_id=detection.tracking_id,
                raw_ocr_output=ocr_result.raw_text[:255],
                parsed_number=ocr_result.parsed_number,
                confidence=ocr_result.confidence,
                is_valid=ocr_result.is_valid,
            )
            self._db.add(jersey_number)
            ocr_count += 1

        return ocr_count

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
            pct, msg = PHASE_SETUP.progress(0.0, "Loading video metadata...", TOTAL_PHASES)
            report_progress(pct, 100, msg)
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

            if self._config.delete_existing:
                pct, msg = PHASE_SETUP.progress(
                    0.5, "Clearing existing detections...", TOTAL_PHASES
                )
                report_progress(pct, 100, msg)
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

            pct, msg = PHASE_SETUP.progress(1.0, "Initializing detector...", TOTAL_PHASES)
            report_progress(pct, 100, msg)
            result = await self._run_detection(video, video_path, report_progress)

            if not result.error and self._config.enable_track_merging:
                pct, msg = PHASE_TRACK_MERGE.progress(
                    0.0, "Merging fragmented tracks...", TOTAL_PHASES
                )
                report_progress(pct, 100, msg)
                merge_result = await self._run_track_merger(video_id)
                if merge_result.error:
                    logger.warning(f"Track merge failed: {merge_result.error}")
                else:
                    logger.info(
                        f"Track merge: {merge_result.original_track_count} → "
                        f"{merge_result.merged_track_count} tracks"
                    )

            if result.error:
                await self._update_video_status(video, ProcessingStatus.FAILED)
            else:
                await self._update_video_status(video, ProcessingStatus.COMPLETED, processed=True)

            pct, msg = PHASE_TRACK_MERGE.progress(1.0, "Complete", TOTAL_PHASES)
            report_progress(pct, 100, msg)
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
        await self._db.execute(delete(JerseyNumber).where(JerseyNumber.video_id == video_id))
        await self._db.execute(delete(PlayerDetection).where(PlayerDetection.video_id == video_id))
        await self._db.commit()
        self._reset_ocr_state()

    async def _run_track_merger(self, video_id: int):
        merger_config = TrackMergerConfig(
            max_temporal_gap_frames=self._config.merge_max_temporal_gap_frames,
            max_spatial_distance=self._config.merge_max_spatial_distance,
            min_size_similarity=self._config.merge_min_size_similarity,
        )
        merger = TrackMerger(self._db, merger_config)
        return await merger.merge_tracks(video_id)

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

            # Calculate end frame if max_seconds is set
            end_frame = None
            if self._config.max_seconds is not None:
                end_frame = int(self._config.max_seconds * metadata.fps)
                end_frame = min(end_frame, metadata.total_frames)

            # Calculate total frames to process
            total_sampled_frames = extractor.count_sampled_frames(
                sample_interval=self._config.sample_interval,
                end_frame=end_frame,
            )

            duration_msg = (
                f" (first {self._config.max_seconds}s)" if self._config.max_seconds else ""
            )
            pct, msg = PHASE_DETECTION.progress(
                0.0, f"Starting {total_sampled_frames} frames{duration_msg}...", TOTAL_PHASES
            )
            report_progress(pct, 100, msg)

            # Process frames in batches
            batch_frames = []
            batch_frame_numbers = []

            for extracted_frame in extractor.extract_frames_sampled(
                sample_interval=self._config.sample_interval,
                end_frame=end_frame,
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

                        person_detections, other_detections = self._filter_persons_only(
                            frame_detections
                        )

                        person_detections = self._extract_colors(person_detections, frame)

                        if tracker:
                            # SAM2 requires frame images for appearance-based tracking
                            if isinstance(tracker, SAM2VideoTracker):
                                person_detections = await asyncio.to_thread(
                                    tracker.update, person_detections, frame
                                )
                            else:
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

                        # 4. Run OCR on legible person detections
                        await self._run_ocr_on_frame(
                            video.id,
                            frame_detections.frame_number,
                            frame,
                            person_detections,
                        )

                        # 5. Combine filtered persons with other detections (balls)
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

                    frame_pct = frames_processed / total_sampled_frames
                    pct, msg = PHASE_DETECTION.progress(
                        frame_pct,
                        f"{frames_processed}/{total_sampled_frames} frames ({int(frame_pct * 100)}%)",
                        TOTAL_PHASES,
                    )
                    report_progress(pct, 100, msg)

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

                for i, frame_detections in enumerate(detections_list):
                    frame = batch_frames[i]

                    person_detections, other_detections = self._filter_persons_only(
                        frame_detections
                    )

                    person_detections = self._extract_colors(person_detections, frame)

                    if tracker:
                        # SAM2 requires frame images for appearance-based tracking
                        if isinstance(tracker, SAM2VideoTracker):
                            person_detections = await asyncio.to_thread(
                                tracker.update, person_detections, frame
                            )
                        else:
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

                    # 4. Run OCR on legible person detections
                    await self._run_ocr_on_frame(
                        video.id,
                        frame_detections.frame_number,
                        frame,
                        person_detections,
                    )

                    # 5. Combine filtered persons with other detections (balls)
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

            pct, msg = PHASE_FINALIZE.progress(0.0, "Committing detections...", TOTAL_PHASES)
            report_progress(pct, 100, msg)

        await self._db.commit()
        pct, msg = PHASE_FINALIZE.progress(
            1.0, f"Saved {total_detections} detections", TOTAL_PHASES
        )
        report_progress(pct, 100, msg)

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

    def _extract_colors(self, frame_detections: FrameDetections, frame) -> FrameDetections:
        enriched = []
        for det in frame_detections.detections:
            color_hist = extract_jersey_color(
                frame, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
            )
            enriched.append(
                Detection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    tracking_id=det.tracking_id,
                    color_hist=color_hist,
                )
            )
        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=enriched,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
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

    This sets up the batch-based orchestrator to work with the background job system.
    The orchestrator checkpoints after each batch for resilience and supports resume.

    When tracking_backend is "sam3", uses SAM3DetectionPipeline which provides
    unified detection and tracking via text-prompted video segmentation.
    """
    from app.database import async_session_maker
    from app.services.job_manager import Job

    async def detection_worker(job: Job, update_progress: Callable[[int, int, str], None]):
        """Worker function for processing video detection jobs using batch orchestrator."""
        video_id = job.metadata.get("video_id")
        if not video_id:
            raise ValueError("video_id required in job metadata")

        async with async_session_maker() as db:
            # Use SAM3 pipeline when sam3 backend is selected
            if settings.tracking_backend == "sam3":
                from pathlib import Path

                from sqlalchemy import delete
                from sqlalchemy.future import select

                from app.models.detection import PlayerDetection
                from app.models.jersey_number import JerseyNumber
                from app.models.processing_batch import ProcessingBatch
                from app.models.video import ProcessingStatus, Video
                from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

                # Get video path
                video_storage_path = Path(settings.video_storage_path)
                result = await db.execute(select(Video).where(Video.id == video_id))
                video = result.scalar_one_or_none()

                if not video:
                    raise ValueError(f"Video not found: {video_id}")

                video_path = video_storage_path / video.file_path
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                # Update status to processing
                video.processing_status = ProcessingStatus.PROCESSING
                await db.commit()

                # Clear existing detections
                await db.execute(delete(JerseyNumber).where(JerseyNumber.video_id == video_id))
                await db.execute(
                    delete(PlayerDetection).where(PlayerDetection.video_id == video_id)
                )
                await db.execute(
                    delete(ProcessingBatch).where(ProcessingBatch.video_id == video_id)
                )
                await db.commit()

                logger.info(f"Starting SAM3 detection job for video {video_id}")
                update_progress(5, 100, "[1/2] SAM3: Processing video...")

                # Create progress callback for SAM3 pipeline
                def on_sam3_progress(current: int, total: int) -> None:
                    if total > 0:
                        pct = 5 + int((current / total) * 90)
                    else:
                        pct = 50  # Unknown total
                    update_progress(pct, 100, f"[1/2] SAM3: Frame {current}...")

                pipeline = SAM3DetectionPipeline(
                    prompt=settings.sam3_prompt,
                    confidence_threshold=settings.sam3_confidence_threshold,
                    on_progress=on_sam3_progress,
                )

                sample_interval = job.metadata.get(
                    "sample_interval", settings.batch_sample_interval
                )
                frames_processed = await pipeline.process_video_to_db(
                    video_id=video_id,
                    video_path=video_path,
                    db_session=db,
                    sample_interval=sample_interval,
                )

                # Get detection count
                from sqlalchemy import func

                count_result = await db.execute(
                    select(func.count())
                    .select_from(PlayerDetection)
                    .where(PlayerDetection.video_id == video_id)
                )
                total_detections = count_result.scalar_one()

                # Update video status
                video.processing_status = ProcessingStatus.COMPLETED
                video.processed = True
                await db.commit()

                update_progress(100, 100, "[2/2] Complete")

                return {
                    "video_id": video_id,
                    "batches_processed": 1,
                    "total_frames_processed": frames_processed,
                    "total_detections": total_detections,
                    "total_ocr_results": 0,  # SAM3 doesn't do OCR yet
                    "resumed_from_batch": None,
                    "tracking_backend": "sam3",
                }

            # Default: use SequentialOrchestrator for bytetrack, norfair, sam2
            resolved_device = SequentialOrchestrator._resolve_device(settings.ml_device)
            config = OrchestratorConfig(
                frames_per_batch=job.metadata.get(
                    "frames_per_batch", settings.batch_frames_per_batch
                ),
                sample_interval=job.metadata.get("sample_interval", settings.batch_sample_interval),
                confidence_threshold=job.metadata.get(
                    "confidence_threshold", settings.yolo_confidence_threshold
                ),
                device=resolved_device,
                max_seconds=job.metadata.get("max_seconds"),
                enable_court_detection=job.metadata.get("enable_court_detection", True),
                enable_jersey_ocr=job.metadata.get("enable_jersey_ocr", True),
                enable_track_merging=job.metadata.get("enable_track_merging", True),
            )
            logger.info(f"Starting detection job for video {video_id} on device: {resolved_device}")
            orchestrator = SequentialOrchestrator(db, config)
            result = await orchestrator.process_video(video_id, update_progress)

            if result.error:
                raise RuntimeError(result.error)

            return {
                "video_id": result.video_id,
                "batches_processed": result.batches_processed,
                "total_frames_processed": result.total_frames,
                "total_detections": result.total_detections,
                "total_ocr_results": result.total_ocr_results,
                "resumed_from_batch": result.resumed_from_batch,
            }

    job_manager.register_worker("video_detection", detection_worker)
