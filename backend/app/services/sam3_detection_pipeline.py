"""SAM3-based detection pipeline for video processing.

This pipeline uses SAM3's unified detection and tracking to process
basketball videos with stable player tracking.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator, Callable

from app.config import settings
from app.ml.sam3_tracker import SAM3TrackerConfig, SAM3VideoTracker
from app.ml.types import FrameDetections

logger = logging.getLogger(__name__)


class SAM3DetectionPipeline:
    """Pipeline for processing videos with SAM3 tracking.

    Replaces the traditional detect -> track pipeline with SAM3's
    unified text-prompted video segmentation.
    """

    def __init__(
        self,
        prompt: str | None = None,
        confidence_threshold: float | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize SAM3 detection pipeline.

        Args:
            prompt: Text prompt for detection (default: from settings).
            confidence_threshold: Min confidence (default: from settings).
            on_progress: Optional callback(current_frame, total_frames).
        """
        config = SAM3TrackerConfig(
            prompt=prompt or settings.sam3_prompt,
            confidence_threshold=confidence_threshold
            or settings.sam3_confidence_threshold,
        )
        self._tracker = SAM3VideoTracker(config)
        self._on_progress = on_progress

    async def process_video(
        self,
        video_path: Path,
        sample_interval: int | None = None,
    ) -> AsyncGenerator[FrameDetections, None]:
        """Process video and yield FrameDetections.

        Args:
            video_path: Path to video file.
            sample_interval: Process every Nth frame (default: from settings).

        Yields:
            FrameDetections for each processed frame.
        """
        # SAM3 requires sample_interval=1 for stable tracking IDs
        interval = sample_interval or settings.sam3_sample_interval

        logger.info(f"Starting SAM3 pipeline for {video_path}")

        frame_count = 0
        for frame_detections in self._tracker.process_video(
            video_path, sample_interval=interval
        ):
            frame_count += 1

            if self._on_progress:
                self._on_progress(frame_count, -1)  # Total unknown

            yield frame_detections

        logger.info(f"SAM3 pipeline complete: {frame_count} frames processed")

    async def process_video_to_db(
        self,
        video_id: int,
        video_path: Path,
        db_session,
        sample_interval: int | None = None,
    ) -> int:
        """Process video and store detections in database.

        Args:
            video_id: Database video ID.
            video_path: Path to video file.
            db_session: Database session for storage.
            sample_interval: Process every Nth frame.

        Returns:
            Number of frames processed.
        """
        from app.models import PlayerDetection

        frame_count = 0

        async for frame_detections in self.process_video(
            video_path, sample_interval=sample_interval
        ):
            # Store each detection in database
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
                db_session.add(detection)

            frame_count += 1

            # Commit every 100 frames for checkpointing
            if frame_count % 100 == 0:
                await db_session.commit()
                logger.debug(f"Checkpointed at frame {frame_count}")

        await db_session.commit()
        return frame_count
