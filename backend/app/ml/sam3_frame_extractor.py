"""Frame extraction utility for SAM3 VideoPredictor.

SAM3's VideoPredictor requires frames as JPEG files on disk.
This module handles extracting frames from video files with
configurable sampling and cleanup.
"""

import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractionResult:
    """Result of frame extraction operation."""

    output_dir: Path
    frame_count: int
    fps: float
    width: int
    height: int
    sample_interval: int
    frame_indices: list[int]  # Original frame numbers that were extracted


class SAM3FrameExtractor:
    """Extracts video frames to JPEG files for SAM3 processing."""

    def __init__(self, jpeg_quality: int = 95):
        """Initialize frame extractor.

        Args:
            jpeg_quality: JPEG compression quality (0-100).
        """
        self.jpeg_quality = jpeg_quality

    def get_video_frame_count(
        self,
        video_path: Path,
        sample_interval: int = 1,
    ) -> int:
        """Get estimated frame count from video metadata.

        This is a fast probe that doesn't read all frames.

        Args:
            video_path: Path to video file.
            sample_interval: If > 1, returns count of sampled frames.

        Returns:
            Estimated number of frames (or sampled frames if interval > 1).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return -1

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return -1
            # Adjust for sample interval
            return (total_frames + sample_interval - 1) // sample_interval
        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        sample_interval: int = 1,
        max_frames: int | None = None,
        start_frame: int = 0,
    ) -> FrameExtractionResult:
        """Extract frames from video to JPEG files.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save JPEG frames.
            sample_interval: Extract every Nth frame (default: every frame).
            max_frames: Maximum number of frames to extract (None = all).
            start_frame: Start extraction from this frame number.

        Returns:
            FrameExtractionResult with metadata about extracted frames.

        Raises:
            ValueError: If video cannot be opened.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Seek to start frame if specified
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_indices = []
            output_idx = 0
            input_idx = start_frame

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check max_frames limit
                if max_frames is not None and output_idx >= max_frames:
                    break

                if (input_idx - start_frame) % sample_interval == 0:
                    # SAM3 expects frames named as 6-digit numbers
                    frame_path = output_dir / f"{output_idx:06d}.jpg"
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    frame_indices.append(input_idx)
                    output_idx += 1

                input_idx += 1

            logger.info(
                f"Extracted {output_idx} frames from {video_path} "
                f"(sample_interval={sample_interval})"
            )

            return FrameExtractionResult(
                output_dir=output_dir,
                frame_count=output_idx,
                fps=fps,
                width=width,
                height=height,
                sample_interval=sample_interval,
                frame_indices=frame_indices,
            )

        finally:
            cap.release()

    @contextmanager
    def temp_frame_folder(
        self, video_id: str
    ) -> Generator[Path, None, None]:
        """Context manager for temporary frame folder with cleanup.

        Args:
            video_id: Unique identifier for the video (used in folder name).

        Yields:
            Path to temporary folder.
        """
        folder = settings.sam3_temp_frames_dir / video_id
        try:
            # Clean up any existing frames from previous crashed runs
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
                logger.debug(f"Cleaned up stale temp frames: {folder}")
            folder.mkdir(parents=True, exist_ok=True)
            yield folder
        finally:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
                logger.debug(f"Cleaned up temp frames: {folder}")
