"""SAM3 video tracker using Meta's Segment Anything Model 3.

SAM3 provides unified detection, segmentation, and tracking with
text prompts. This module wraps SAM3's VideoModel for basketball
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

    Uses SAM3's VideoModel to detect and track all instances of
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
        self._model = None
        self._processor = None
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
        """Lazy load SAM3 video model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            self._device = self._select_device()
            dtype = torch.bfloat16 if self._config.use_half_precision else torch.float32

            logger.info(f"Loading SAM3 video model on {self._device}...")

            self._model = Sam3VideoModel.from_pretrained(
                "facebook/sam3",
                torch_dtype=dtype,
            ).to(self._device)

            self._processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

            logger.info("SAM3 video model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "SAM3 not installed. Install with: "
                "uv pip install transformers>=4.48"
            ) from e

    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 1,
        max_frames: int | None = None,
        start_frame: int = 0,
    ) -> Generator[FrameDetections, None, None]:
        """Process video and yield FrameDetections for each frame.

        SAM3's video tracking relies on temporal continuity between frames.
        Using sample_interval > 1 will cause tracking IDs to become unstable
        as players move significantly between sampled frames.

        Args:
            video_path: Path to input video file.
            sample_interval: Process every Nth frame. Default 1 (every frame)
                is strongly recommended for stable tracking IDs.
            max_frames: Maximum number of frames to process (None = all).
            start_frame: Start processing from this frame number.

        Yields:
            FrameDetections for each processed frame with stable tracking IDs.
        """
        from PIL import Image

        self._load_predictor()

        video_id = video_path.stem

        with self._frame_extractor.temp_frame_folder(video_id) as frames_dir:
            # Extract frames to JPEG folder
            extraction = self._frame_extractor.extract_frames(
                video_path,
                frames_dir,
                sample_interval=sample_interval,
                max_frames=max_frames,
                start_frame=start_frame,
            )

            logger.info(
                f"Processing {extraction.frame_count} frames with SAM3 "
                f"(prompt='{self._config.prompt}')"
            )

            # Load all frames as PIL images
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            if not frame_paths:
                logger.warning("No frames extracted")
                return

            # Load frames as list of PIL images for the processor
            video_frames = [Image.open(fp) for fp in frame_paths]

            # Get dtype for model
            dtype = torch.bfloat16 if self._config.use_half_precision else torch.float32

            # Initialize video inference session
            inference_session = self._processor.init_video_session(
                video=video_frames,
                inference_device=self._device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype,
            )

            # Add text prompt for detection and tracking
            inference_session = self._processor.add_text_prompt(
                inference_session=inference_session,
                text=self._config.prompt,
            )

            logger.info(f"Added text prompt: '{self._config.prompt}'")

            # Process all frames using propagate_in_video_iterator
            outputs_per_frame = {}
            for model_outputs in self._model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(video_frames),
            ):
                processed_outputs = self._processor.postprocess_outputs(
                    inference_session, model_outputs
                )
                outputs_per_frame[model_outputs.frame_idx] = processed_outputs

            logger.info(f"Processed {len(outputs_per_frame)} frames")

            # Yield detections for each frame
            for frame_idx in sorted(outputs_per_frame.keys()):
                frame_outputs = outputs_per_frame[frame_idx]
                original_frame_number = extraction.frame_indices[frame_idx]

                frame_detections = self._convert_to_frame_detections(
                    frame_outputs,
                    frame_number=original_frame_number,
                    frame_width=extraction.width,
                    frame_height=extraction.height,
                )
                yield frame_detections

    def _convert_to_frame_detections(
        self,
        sam3_output: dict,
        frame_number: int,
        frame_width: int,
        frame_height: int,
    ) -> FrameDetections:
        """Convert SAM3 output to FrameDetections type.

        Args:
            sam3_output: Output dict from SAM3 postprocess_outputs.
            frame_number: Original video frame number.
            frame_width: Video frame width.
            frame_height: Video frame height.

        Returns:
            FrameDetections compatible with existing pipeline.
        """
        detections = []

        # Extract arrays from output
        object_ids = sam3_output.get("object_ids", np.array([]))
        boxes = sam3_output.get("boxes", np.array([]))
        scores = sam3_output.get("scores", np.array([]))

        # Convert to numpy if tensors (handle GPU tensors by moving to CPU first)
        if hasattr(object_ids, "cpu"):
            object_ids = object_ids.cpu().numpy()
        elif hasattr(object_ids, "numpy"):
            object_ids = object_ids.numpy()
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        elif hasattr(boxes, "numpy"):
            boxes = boxes.numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        elif hasattr(scores, "numpy"):
            scores = scores.numpy()

        for i, obj_id in enumerate(object_ids):
            # Skip low confidence detections
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if confidence < self._config.confidence_threshold:
                continue

            # Convert box from xyxy to BoundingBox
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
