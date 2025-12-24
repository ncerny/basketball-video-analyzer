"""SAM2 tracker using Meta's Segment Anything 2 for mask-enhanced tracking.

SAM2 differs from traditional trackers (ByteTrack, Norfair):
- Uses SAM2 image predictor for precise segmentation masks
- Masks enable better IOU matching than bbox-only approaches
- Supports mask-based re-identification across frames

This tracker uses SAM2's image predictor (not video predictor) for frame-by-frame
processing with custom mask-based tracking logic.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config import settings

from .types import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class SAM2TrackerConfig:
    """Configuration for SAM2 tracking."""

    model_name: str = "sam2_hiera_tiny"
    device: str = "mps"

    # Matching thresholds
    min_iou_threshold: float = 0.3  # Minimum IOU to consider a match
    lost_track_frames: int = 30  # Frames before dropping lost track

    # Mask quality settings
    min_mask_area: int = 100  # Minimum mask area to consider valid


@dataclass
class _TrackedObject:
    """Internal state for a tracked object."""

    track_id: int
    last_mask: np.ndarray | None = None
    last_bbox: BoundingBox | None = None
    last_seen_frame: int = 0
    hit_count: int = 0


class SAM2VideoTracker:
    """SAM2-based tracker using image predictor for frame-by-frame segmentation.

    Uses SAM2 to generate precise segmentation masks from RF-DETR bounding boxes,
    then matches masks across frames using IOU for tracking.

    Requires frame images in addition to detections - call with frame parameter.

    Example:
        tracker = SAM2VideoTracker()
        tracked = tracker.update(frame_detections, frame)  # frame is np.ndarray
    """

    def __init__(self, config: SAM2TrackerConfig | None = None) -> None:
        """Initialize SAM2 tracker.

        Args:
            config: Tracker configuration. If None, uses defaults from settings.
        """
        self._config = config or SAM2TrackerConfig(
            model_name=settings.sam2_model_name,
            min_iou_threshold=settings.sam2_new_object_iou_threshold,
            lost_track_frames=settings.sam2_lost_track_frames,
        )

        # Lazy-loaded components
        self._model = None
        self._predictor = None
        self._device = None

        # Track management
        self._next_track_id = 1
        self._tracks: dict[int, _TrackedObject] = {}  # track_id -> TrackedObject
        self._frame_count = 0

        logger.info(
            f"SAM2VideoTracker initialized with model={self._config.model_name}, "
            f"device={self._config.device}"
        )

    def _load_model(self) -> None:
        """Lazy load SAM2 image predictor."""
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Map model name to config file
            model_configs = {
                "sam2_hiera_tiny": "sam2_hiera_t.yaml",
                "sam2_hiera_small": "sam2_hiera_s.yaml",
                "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
                "sam2_hiera_large": "sam2_hiera_l.yaml",
            }

            config_name = model_configs.get(self._config.model_name)
            if config_name is None:
                raise ValueError(f"Unknown SAM2 model: {self._config.model_name}")

            checkpoint_path = settings.models_dir / f"{self._config.model_name}.pt"

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {checkpoint_path}. "
                    f"Run: python -m scripts.download_sam2_model {self._config.model_name}"
                )

            logger.info(f"Loading SAM2 model from {checkpoint_path}...")

            # Handle device selection
            self._device = self._config.device
            if self._device == "auto":
                import torch

                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            self._model = build_sam2(
                config_name,
                str(checkpoint_path),
                device=self._device,
            )
            self._predictor = SAM2ImagePredictor(self._model)

            logger.info(f"SAM2 model loaded successfully on {self._device}")

        except ImportError as e:
            raise ImportError(
                "SAM2 not installed. Install with: pip install sam2"
            ) from e

    def update(
        self,
        frame_detections: FrameDetections,
        frame: np.ndarray | None = None,
    ) -> FrameDetections:
        """Update tracker with new frame and detections.

        Args:
            frame_detections: Detections from RF-DETR for current frame.
            frame: The actual frame image (BGR numpy array). Required for SAM2.

        Returns:
            FrameDetections with tracking_id assigned to each detection.

        Raises:
            ValueError: If frame is not provided.
        """
        if frame is None:
            raise ValueError(
                "SAM2 tracker requires frame image. "
                "Pass frame parameter: tracker.update(detections, frame)"
            )

        self._load_model()
        self._frame_count += 1

        # Set image for SAM2 predictor (converts BGR to RGB internally)
        frame_rgb = frame[:, :, ::-1].copy()
        self._predictor.set_image(frame_rgb)

        # Generate masks for all detections using batch prediction
        detection_masks = self._generate_masks_batch(frame_detections)

        # Match detections to existing tracks
        matched_detections = self._match_and_update_tracks(
            frame_detections, detection_masks
        )

        # Clean up lost tracks
        self._cleanup_lost_tracks()

        return matched_detections

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._next_track_id = 1
        self._tracks.clear()
        self._frame_count = 0
        logger.debug("SAM2 tracker reset")

    def _generate_masks_batch(
        self, frame_detections: FrameDetections
    ) -> list[np.ndarray | None]:
        """Generate SAM2 masks for all detections in batch.

        Args:
            frame_detections: Detections to segment.

        Returns:
            List of masks (or None for failed segmentations).
        """
        if not frame_detections.detections:
            return []

        # Prepare batch of bounding boxes
        boxes = []
        for det in frame_detections.detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            boxes.append([x1, y1, x2, y2])

        boxes_array = np.array(boxes, dtype=np.float32)

        try:
            # Batch predict masks for all boxes
            masks, scores, _ = self._predictor.predict(
                box=boxes_array,
                multimask_output=False,  # Single best mask per box
            )

            # Extract best mask for each detection
            result_masks = []
            for i in range(len(frame_detections.detections)):
                if i < len(masks):
                    mask = masks[i].squeeze()
                    if mask.sum() >= self._config.min_mask_area:
                        result_masks.append(mask.astype(bool))
                    else:
                        result_masks.append(None)
                else:
                    result_masks.append(None)

            return result_masks

        except Exception as e:
            logger.warning(f"SAM2 batch prediction failed: {e}")
            return [None] * len(frame_detections.detections)

    def _match_and_update_tracks(
        self,
        frame_detections: FrameDetections,
        detection_masks: list[np.ndarray | None],
    ) -> FrameDetections:
        """Match detections to existing tracks and update track states.

        Args:
            frame_detections: New detections.
            detection_masks: Corresponding SAM2 masks.

        Returns:
            FrameDetections with tracking IDs assigned.
        """
        tracked_detections = []
        used_track_ids = set()

        # Build cost matrix: IOU between each detection and existing tracks
        for i, det in enumerate(frame_detections.detections):
            det_mask = detection_masks[i] if i < len(detection_masks) else None
            det_bbox = det.bbox

            best_track_id = None
            best_iou = 0.0

            # Compare against existing tracks
            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue

                # Compute IOU - prefer mask-to-mask, fallback to bbox
                if det_mask is not None and track.last_mask is not None:
                    iou = self._compute_mask_iou(det_mask, track.last_mask)
                elif track.last_bbox is not None:
                    iou = self._compute_bbox_iou(det_bbox, track.last_bbox)
                else:
                    iou = 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            # Decide: match existing track or create new
            if best_iou >= self._config.min_iou_threshold and best_track_id is not None:
                track_id = best_track_id
                used_track_ids.add(track_id)
                # Update existing track
                self._tracks[track_id].last_mask = det_mask
                self._tracks[track_id].last_bbox = det_bbox
                self._tracks[track_id].last_seen_frame = self._frame_count
                self._tracks[track_id].hit_count += 1
            else:
                # Create new track
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _TrackedObject(
                    track_id=track_id,
                    last_mask=det_mask,
                    last_bbox=det_bbox,
                    last_seen_frame=self._frame_count,
                    hit_count=1,
                )

            tracked_detections.append(
                Detection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    tracking_id=track_id,
                    color_hist=det.color_hist,
                    shoe_color_hist=det.shoe_color_hist,
                )
            )

        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=tracked_detections,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
        )

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IOU between two binary masks.

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.

        Returns:
            IOU value between 0 and 1.
        """
        # Ensure same shape
        if mask1.shape != mask2.shape:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        return float(intersection / union) if union > 0 else 0.0

    def _compute_bbox_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute IOU between two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.

        Returns:
            IOU value between 0 and 1.
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = bbox2.to_xyxy()

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0

    def _cleanup_lost_tracks(self) -> None:
        """Remove tracks not seen for too many frames."""
        lost_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if self._frame_count - track.last_seen_frame > self._config.lost_track_frames
        ]

        for track_id in lost_track_ids:
            del self._tracks[track_id]
            logger.debug(f"Removed lost track: {track_id}")

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count
