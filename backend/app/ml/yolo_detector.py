"""YOLOv8 detector implementation for player detection."""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector
from .types import BoundingBox, Detection, DetectionClass, FrameDetections


class YOLODetector(BaseDetector):
    """YOLOv8-based detector for player and ball detection.

    Uses YOLOv8-nano by default for lightweight, local-friendly inference.
    Filters detections to only include persons and sports balls.
    """

    # COCO class IDs for filtering
    PERSON_CLASS_ID = DetectionClass.PERSON.value  # 0
    SPORTS_BALL_CLASS_ID = DetectionClass.SPORTS_BALL.value  # 32
    VALID_CLASS_IDS = {PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID}

    # Map COCO class IDs to human-readable names
    CLASS_NAMES = {
        PERSON_CLASS_ID: "person",
        SPORTS_BALL_CLASS_ID: "sports_ball",
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO model weights. If None, downloads yolov8n.pt.
            confidence_threshold: Minimum confidence score for detections (0-1).
            device: Device to run inference on ('cpu', 'cuda', 'mps').
        """
        self._model_path = model_path or "yolov8n.pt"
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: YOLO | None = None

    def _load_model(self) -> None:
        """Load the YOLO model if not already loaded."""
        if self._model is None:
            model = YOLO(self._model_path)
            # Set device for inference
            model.to(self._device)
            self._model = model

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model is not None

    def get_model_info(self) -> dict[str, str | int | float]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata.
        """
        self._load_model()
        assert self._model is not None

        return {
            "model_path": str(self._model_path),
            "device": self._device,
            "confidence_threshold": self._confidence_threshold,
            "model_type": self._model.type if hasattr(self._model, "type") else "unknown",
            "task": str(self._model.task) if hasattr(self._model, "task") else "detect",
        }

    def detect(self, frame: Any, frame_number: int = 0) -> FrameDetections:
        """Detect players and ball in a single frame.

        Args:
            frame: Input frame as numpy array (HxWxC, BGR format from OpenCV).
            frame_number: Frame number for tracking context.

        Returns:
            FrameDetections containing all detected players and balls.
        """
        self._load_model()
        assert self._model is not None

        # Run inference
        results = self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            verbose=False,
            device=self._device,
        )

        # Process results
        detections = self._process_results(results[0]) if results else []

        # Get frame dimensions
        if isinstance(frame, np.ndarray):
            height, width = frame.shape[:2]
        else:
            height, width = 0, 0

        return FrameDetections(
            frame_number=frame_number,
            detections=detections,
            frame_width=width,
            frame_height=height,
        )

    def detect_batch(self, frames: list[Any], start_frame_number: int = 0) -> list[FrameDetections]:
        """Detect players and ball in a batch of frames.

        Args:
            frames: List of input frames as numpy arrays.
            start_frame_number: Frame number of the first frame.

        Returns:
            List of FrameDetections, one per input frame.
        """
        self._load_model()
        assert self._model is not None

        if not frames:
            return []

        # Run batch inference
        results = self._model.predict(
            source=frames,
            conf=self._confidence_threshold,
            verbose=False,
            device=self._device,
        )

        # Process each result
        frame_detections = []
        for i, result in enumerate(results):
            detections = self._process_results(result)

            # Get frame dimensions
            frame = frames[i]
            if isinstance(frame, np.ndarray):
                height, width = frame.shape[:2]
            else:
                height, width = 0, 0

            frame_detections.append(
                FrameDetections(
                    frame_number=start_frame_number + i,
                    detections=detections,
                    frame_width=width,
                    frame_height=height,
                )
            )

        return frame_detections

    def _process_results(self, result: Any) -> list[Detection]:
        """Process YOLO results and filter for valid classes.

        Args:
            result: Single YOLO result object.

        Returns:
            List of Detection objects for valid classes.
        """
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            # Get class ID
            class_id = int(boxes.cls[i].item())

            # Filter to only valid classes (person, sports_ball)
            if class_id not in self.VALID_CLASS_IDS:
                continue

            # Get bounding box in xyxy format
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy

            # Get confidence
            confidence = float(boxes.conf[i].item())

            # Create detection
            detection = Detection(
                bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
                confidence=confidence,
                class_id=class_id,
                class_name=self.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                tracking_id=None,  # Tracking IDs assigned by separate tracker
            )
            detections.append(detection)

        return detections

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for detections.

        Args:
            threshold: New confidence threshold (0-1).
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self._confidence_threshold = threshold

    @property
    def confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self._confidence_threshold

    @property
    def device(self) -> str:
        """Get the device being used for inference."""
        return self._device
