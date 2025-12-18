"""Video frame extraction service using OpenCV."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float

    @property
    def frame_duration_ms(self) -> float:
        """Duration of a single frame in milliseconds."""
        return 1000.0 / self.fps if self.fps > 0 else 0.0


@dataclass
class ExtractedFrame:
    """A single extracted frame with metadata."""

    frame: np.ndarray
    frame_number: int
    timestamp_ms: float


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""

    pass


class FrameExtractor:
    """Service for extracting frames from video files.

    Supports various extraction modes:
    - Extract every Nth frame for efficiency
    - Extract frames at specific timestamps
    - Extract a range of frames
    - Generator-based extraction for memory efficiency
    """

    def __init__(self, video_path: str | Path) -> None:
        """Initialize the frame extractor.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            FrameExtractionError: If video cannot be opened.
        """
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def _open_video(self) -> cv2.VideoCapture:
        """Open video capture if not already open."""
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise FrameExtractionError(f"Failed to open video: {self._video_path}")
        return self._cap

    def _close_video(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameExtractor":
        """Context manager entry."""
        self._open_video()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self._close_video()

    def get_metadata(self) -> VideoMetadata:
        """Get video metadata.

        Returns:
            VideoMetadata with video properties.

        Raises:
            FrameExtractionError: If metadata cannot be extracted.
        """
        if self._metadata is not None:
            return self._metadata

        cap = self._open_video()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30.0  # Default fallback

        duration = total_frames / fps if fps > 0 else 0.0

        self._metadata = VideoMetadata(
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration,
        )

        return self._metadata

    def extract_frame(self, frame_number: int) -> ExtractedFrame:
        """Extract a single frame by frame number.

        Args:
            frame_number: Zero-based frame index.

        Returns:
            ExtractedFrame with the frame data.

        Raises:
            FrameExtractionError: If frame cannot be extracted.
            ValueError: If frame_number is out of range.
        """
        metadata = self.get_metadata()

        if frame_number < 0 or frame_number >= metadata.total_frames:
            raise ValueError(
                f"Frame number {frame_number} out of range [0, {metadata.total_frames - 1}]"
            )

        cap = self._open_video()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            raise FrameExtractionError(f"Failed to read frame {frame_number}")

        timestamp_ms = frame_number * metadata.frame_duration_ms

        return ExtractedFrame(
            frame=frame,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
        )

    def extract_frame_at_timestamp(self, timestamp_ms: float) -> ExtractedFrame:
        """Extract frame at a specific timestamp.

        Args:
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            ExtractedFrame closest to the requested timestamp.

        Raises:
            ValueError: If timestamp is out of range.
            FrameExtractionError: If frame cannot be extracted.
        """
        metadata = self.get_metadata()

        if timestamp_ms < 0 or timestamp_ms > metadata.duration_seconds * 1000:
            raise ValueError(
                f"Timestamp {timestamp_ms}ms out of range [0, {metadata.duration_seconds * 1000}ms]"
            )

        frame_number = int(timestamp_ms / metadata.frame_duration_ms)
        frame_number = min(frame_number, metadata.total_frames - 1)

        return self.extract_frame(frame_number)

    def extract_frames_sampled(
        self,
        sample_interval: int = 3,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Generator[ExtractedFrame, None, None]:
        """Extract frames at regular intervals (generator).

        Memory-efficient extraction for processing pipelines.

        Args:
            sample_interval: Extract every Nth frame (default: 3).
            start_frame: Starting frame number (default: 0).
            end_frame: Ending frame number, exclusive (default: all frames).

        Yields:
            ExtractedFrame for each sampled frame.

        Raises:
            ValueError: If parameters are invalid.
            FrameExtractionError: If extraction fails.
        """
        if sample_interval < 1:
            raise ValueError("sample_interval must be at least 1")

        metadata = self.get_metadata()

        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")

        if end_frame is None:
            end_frame = metadata.total_frames
        elif end_frame > metadata.total_frames:
            end_frame = metadata.total_frames

        if start_frame >= end_frame:
            return

        cap = self._open_video()

        for frame_num in range(start_frame, end_frame, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                # Skip failed frames rather than raising
                continue

            timestamp_ms = frame_num * metadata.frame_duration_ms

            yield ExtractedFrame(
                frame=frame,
                frame_number=frame_num,
                timestamp_ms=timestamp_ms,
            )

    def extract_frames_batch(
        self,
        frame_numbers: list[int],
    ) -> list[ExtractedFrame]:
        """Extract specific frames by frame numbers.

        More efficient than individual calls when extracting multiple frames.

        Args:
            frame_numbers: List of frame numbers to extract.

        Returns:
            List of ExtractedFrame objects (may be shorter if some frames fail).
        """
        if not frame_numbers:
            return []

        metadata = self.get_metadata()
        cap = self._open_video()

        # Sort for efficient sequential access
        sorted_frames = sorted(set(frame_numbers))
        results = []

        for frame_num in sorted_frames:
            if frame_num < 0 or frame_num >= metadata.total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            timestamp_ms = frame_num * metadata.frame_duration_ms

            results.append(
                ExtractedFrame(
                    frame=frame,
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                )
            )

        return results

    def extract_frames_range(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Generator[ExtractedFrame, None, None]:
        """Extract all frames in a range (generator).

        Args:
            start_frame: Starting frame number (default: 0).
            end_frame: Ending frame number, exclusive (default: all frames).

        Yields:
            ExtractedFrame for each frame in range.
        """
        return self.extract_frames_sampled(
            sample_interval=1,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def count_sampled_frames(
        self,
        sample_interval: int = 3,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> int:
        """Count how many frames would be extracted with given parameters.

        Useful for progress tracking without actually extracting frames.

        Args:
            sample_interval: Extract every Nth frame.
            start_frame: Starting frame number.
            end_frame: Ending frame number, exclusive.

        Returns:
            Number of frames that would be extracted.
        """
        metadata = self.get_metadata()

        if end_frame is None:
            end_frame = metadata.total_frames

        if start_frame >= end_frame:
            return 0

        return len(range(start_frame, end_frame, sample_interval))

    def close(self) -> None:
        """Release video capture resources.

        Can be called explicitly, or use context manager instead.
        """
        self._close_video()
