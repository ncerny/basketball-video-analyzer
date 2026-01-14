"""SAM3 video tracker using Meta's Segment Anything Model 3.

SAM3 provides unified detection, segmentation, and tracking with
text prompts. This module wraps SAM3's VideoPredictor for basketball
player tracking.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from app.config import settings

from .sam3_frame_extractor import SAM3FrameExtractor
from .types import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class SAM3TrackerConfig:
    """Configuration for SAM3 video tracker."""

    prompt: str = "basketball player"
    confidence_threshold: float = 0.25
    device: str = "auto"
    use_half_precision: bool = True


class SAM3VideoTracker:
    """SAM3-based tracker using text-prompted video segmentation.

    Uses SAM3's VideoPredictor to detect and track all instances of
    "basketball player" (or custom prompt) throughout a video with
    stable object IDs.

    Example:
        tracker = SAM3VideoTracker(SAM3TrackerConfig())
        for frame_detections in tracker.process_video(video_path):
            # Each detection has stable tracking_id
            print(frame_detections)
    """

    def __init__(self, config: SAM3TrackerConfig | None = None) -> None:
        """Initialize SAM3 tracker.

        Args:
            config: Tracker configuration. If None, uses defaults from settings.
        """
        self._config = config or SAM3TrackerConfig(
            prompt=settings.sam3_prompt,
            confidence_threshold=settings.sam3_confidence_threshold,
            use_half_precision=settings.sam3_use_half_precision,
        )

        # Lazy-loaded components
        self._predictor = None
        self._device = None
        self._frame_extractor = SAM3FrameExtractor()

        logger.info(
            f"SAM3VideoTracker initialized with prompt='{self._config.prompt}', "
            f"device={self._config.device}"
        )

    def _select_device(self) -> str:
        """Select best available device with fallback chain.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'.
        """
        if self._config.device != "auto":
            return self._config.device

        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"

        if torch.backends.mps.is_available():
            # Test MPS actually works
            try:
                torch.zeros(1, device="mps")
                logger.info("Using MPS device")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS available but failed test: {e}")

        logger.info("Using CPU device")
        return "cpu"

    def _load_predictor(self) -> None:
        """Lazy load SAM3 VideoPredictor."""
        if self._predictor is not None:
            return

        try:
            from transformers import Sam3VideoPredictor

            self._device = self._select_device()

            logger.info(f"Loading SAM3 VideoPredictor on {self._device}...")

            self._predictor = Sam3VideoPredictor.from_pretrained(
                "facebook/sam3",
                device=self._device,
                torch_dtype="float16" if self._config.use_half_precision else "float32",
            )

            logger.info("SAM3 VideoPredictor loaded successfully")

        except ImportError as e:
            raise ImportError(
                "SAM3 not installed. Install with: "
                "pip install git+https://github.com/huggingface/transformers"
            ) from e

    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 3,
    ) -> Generator[FrameDetections, None, None]:
        """Process video and yield FrameDetections for each frame.

        Args:
            video_path: Path to input video file.
            sample_interval: Process every Nth frame.

        Yields:
            FrameDetections for each processed frame with stable tracking IDs.
        """
        self._load_predictor()

        video_id = video_path.stem

        with self._frame_extractor.temp_frame_folder(video_id) as frames_dir:
            # Extract frames to JPEG folder
            extraction = self._frame_extractor.extract_frames(
                video_path, frames_dir, sample_interval=sample_interval
            )

            logger.info(
                f"Processing {extraction.frame_count} frames with SAM3 "
                f"(prompt='{self._config.prompt}')"
            )

            # Start SAM3 session
            session_id = self._predictor.start_session(str(frames_dir))

            try:
                # Add text prompt at frame 0
                self._predictor.add_prompt(
                    session_id=session_id,
                    frame_index=0,
                    text=self._config.prompt,
                )

                # Propagate through video
                for output in self._predictor.propagate_in_video(
                    session_id=session_id,
                    direction="forward",
                ):
                    # Map SAM3 frame index back to original video frame number
                    sam3_frame_idx = output["frame_index"]
                    original_frame_number = extraction.frame_indices[sam3_frame_idx]

                    frame_detections = self._convert_to_frame_detections(
                        output,
                        frame_number=original_frame_number,
                        frame_width=extraction.width,
                        frame_height=extraction.height,
                    )

                    yield frame_detections

            finally:
                self._predictor.close_session(session_id)

    def _convert_to_frame_detections(
        self,
        sam3_output: dict,
        frame_number: int,
        frame_width: int,
        frame_height: int,
    ) -> FrameDetections:
        """Convert SAM3 output to FrameDetections type.

        Args:
            sam3_output: Output dict from SAM3 propagate_in_video.
            frame_number: Original video frame number.
            frame_width: Video frame width.
            frame_height: Video frame height.

        Returns:
            FrameDetections compatible with existing pipeline.
        """
        detections = []

        object_ids = sam3_output.get("object_ids", [])
        boxes = sam3_output.get("boxes", np.array([]))
        scores = sam3_output.get("scores", np.array([]))
        masks = sam3_output.get("masks", [])

        for i, obj_id in enumerate(object_ids):
            # Skip low confidence detections
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if confidence < self._config.confidence_threshold:
                continue

            # Convert box from x1y1x2y2 to BoundingBox
            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                bbox = BoundingBox.from_xyxy(float(x1), float(y1), float(x2), float(y2))
            else:
                continue

            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=0,  # person
                class_name="person",
                tracking_id=int(obj_id),
            )
            detections.append(detection)

        return FrameDetections(
            frame_number=frame_number,
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
