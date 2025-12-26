"""SAM2 tracker using Meta's Segment Anything 2 for mask-enhanced tracking.

SAM2 differs from traditional trackers (ByteTrack, Norfair):
- Uses SAM2 image predictor for precise segmentation masks
- Masks enable better IOU matching than bbox-only approaches
- Supports mask-based re-identification across frames
- Embedding-based re-identification for robust tracking through occlusions

This tracker uses SAM2's image predictor (not video predictor) for frame-by-frame
processing with custom mask-based tracking logic.
"""

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import settings

from .types import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)

# SAM 2.1 checkpoint download URLs from Meta (092824 release)
# SAM 2.1 has improved occlusion handling and better tracking of similar objects
# Source: https://github.com/facebookresearch/sam2
SAM2_CHECKPOINT_URLS = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


def download_sam2_checkpoint(model_name: str, target_path: Path) -> None:
    """Download SAM2 checkpoint from Meta's servers.

    Args:
        model_name: Name of the model (e.g., 'sam2_hiera_tiny').
        target_path: Path where the checkpoint should be saved.

    Raises:
        ValueError: If model_name is not recognized.
        urllib.error.URLError: If download fails.
    """
    if model_name not in SAM2_CHECKPOINT_URLS:
        raise ValueError(
            f"Unknown SAM2 model: {model_name}. "
            f"Available: {list(SAM2_CHECKPOINT_URLS.keys())}"
        )

    url = SAM2_CHECKPOINT_URLS[model_name]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading SAM2 checkpoint: {model_name} from {url}")
    logger.info(f"Target path: {target_path}")

    # Download with progress reporting
    def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            if block_num % 100 == 0:  # Log every 100 blocks
                logger.info(f"Download progress: {percent}%")

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=_report_progress)
        logger.info(f"Successfully downloaded {model_name} to {target_path}")
    except Exception as e:
        # Clean up partial download
        if target_path.exists():
            target_path.unlink()
        raise


@dataclass
class SAM2TrackerConfig:
    """Configuration for SAM2 tracking.

    Defaults match global settings (sam2_* in config.py).
    """

    model_name: str = "sam2_hiera_tiny"
    device: str = "auto"

    # Matching thresholds
    min_iou_threshold: float = 0.3  # Minimum IOU to consider a match
    lost_track_frames: int = 0  # 0 = keep tracks for entire video

    # Mask quality settings
    min_mask_area: int = 100  # Minimum mask area to consider valid

    # Embedding-based re-identification
    embedding_similarity_threshold: float = 0.5  # Min cosine similarity for match
    color_tiebreaker_threshold: float = 0.15  # Use color when scores within this
    reidentification_enabled: bool = True  # Enable embedding-based re-ID


@dataclass
class _TrackedObject:
    """Internal state for a tracked object."""

    track_id: int
    last_mask: np.ndarray | None = None
    last_bbox: BoundingBox | None = None
    last_seen_frame: int = 0
    hit_count: int = 0
    # Embedding for re-identification (extracted from SAM2 encoder)
    embedding: np.ndarray | None = None
    # Color histogram for tiebreaking
    color_hist: np.ndarray | None = None


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
            embedding_similarity_threshold=settings.sam2_embedding_similarity_threshold,
            color_tiebreaker_threshold=settings.sam2_color_tiebreaker_threshold,
            reidentification_enabled=settings.sam2_reidentification_enabled,
        )

        # Lazy-loaded components
        self._model = None
        self._predictor = None
        self._device = None

        # Track management
        self._next_track_id = 1
        self._tracks: dict[int, _TrackedObject] = {}  # track_id -> TrackedObject
        self._frame_count = 0

        # Cache for current frame's image features (from SAM2 encoder)
        self._current_frame_features: np.ndarray | None = None

        logger.info(
            f"SAM2VideoTracker initialized with model={self._config.model_name}, "
            f"device={self._config.device}, reidentification={self._config.reidentification_enabled}"
        )

    def _load_model(self) -> None:
        """Lazy load SAM2 image predictor."""
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Map model name to config file
            # SAM 2.1 config files (in sam2.1/ subdirectory)
            model_configs = {
                "sam2_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            }

            config_name = model_configs.get(self._config.model_name)
            if config_name is None:
                raise ValueError(f"Unknown SAM2 model: {self._config.model_name}")

            checkpoint_path = settings.models_dir / f"{self._config.model_name}.pt"

            if not checkpoint_path.exists():
                if settings.sam2_auto_download:
                    logger.info(f"SAM2 checkpoint not found, auto-downloading...")
                    download_sam2_checkpoint(self._config.model_name, checkpoint_path)
                else:
                    raise FileNotFoundError(
                        f"SAM2 checkpoint not found: {checkpoint_path}. "
                        f"Enable sam2_auto_download or run: "
                        f"python -m scripts.download_sam2_model {self._config.model_name}"
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

        # Extract embeddings for all detections (for re-identification)
        detection_embeddings = self._extract_embeddings_batch(frame_detections, detection_masks)

        # Match detections to existing tracks using embeddings + IOU + color
        matched_detections = self._match_and_update_tracks(
            frame_detections, detection_masks, detection_embeddings
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

    def _extract_embeddings_batch(
        self, frame_detections: FrameDetections, masks: list[np.ndarray | None]
    ) -> list[np.ndarray | None]:
        """Extract embeddings for all detections from SAM2's image features.

        Uses the image encoder features computed during set_image() and pools
        them within each detection's mask region to create a compact embedding.

        Args:
            frame_detections: Detections to extract embeddings for.
            masks: Corresponding SAM2 masks (used for masked pooling).

        Returns:
            List of embedding vectors (or None if extraction failed).
        """
        if not self._config.reidentification_enabled:
            return [None] * len(frame_detections.detections)

        embeddings = []

        try:
            # Access SAM2's internal image features
            # After set_image(), features are stored in predictor._features
            if not hasattr(self._predictor, "_features") or self._predictor._features is None:
                logger.warning("No image features available for embedding extraction")
                return [None] * len(frame_detections.detections)

            # Debug: log available feature keys
            if self._frame_count == 1:
                logger.info(f"SAM2 feature keys: {list(self._predictor._features.keys())}")

            # Get the high-resolution feature map from SAM2's encoder
            # SAM2 stores features differently - check available keys
            if "high_res_feats" in self._predictor._features:
                features = self._predictor._features["high_res_feats"][-1]
            elif "image_embed" in self._predictor._features:
                features = self._predictor._features["image_embed"]
            else:
                logger.warning(f"Unknown feature structure: {list(self._predictor._features.keys())}")
                return [None] * len(frame_detections.detections)

            if hasattr(features, "cpu"):
                features = features.cpu().numpy()

            # features shape: (1, C, H, W)
            _, feat_dim, feat_h, feat_w = features.shape

            for i, det in enumerate(frame_detections.detections):
                mask = masks[i] if i < len(masks) else None

                if mask is None:
                    embeddings.append(None)
                    continue

                try:
                    # Get bbox for ROI extraction
                    x1, y1, x2, y2 = det.bbox.to_xyxy()

                    # Scale bbox to feature map coordinates
                    # _orig_hw is a list of (H, W) tuples - we need the first element
                    orig_hw = self._predictor._orig_hw[0]
                    scale_x = feat_w / orig_hw[1]  # width
                    scale_y = feat_h / orig_hw[0]  # height

                    fx1 = max(0, int(x1 * scale_x))
                    fy1 = max(0, int(y1 * scale_y))
                    fx2 = min(feat_w, int(x2 * scale_x) + 1)
                    fy2 = min(feat_h, int(y2 * scale_y) + 1)

                    if fx2 <= fx1 or fy2 <= fy1:
                        embeddings.append(None)
                        continue

                    # Extract ROI features and pool
                    roi_features = features[0, :, fy1:fy2, fx1:fx2]  # (C, h, w)

                    # Global average pooling over spatial dimensions
                    embedding = roi_features.mean(axis=(1, 2))  # (C,)

                    # L2 normalize for cosine similarity
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    embeddings.append(embedding.astype(np.float32))

                except Exception as e:
                    logger.debug(f"Failed to extract embedding for detection {i}: {e}")
                    embeddings.append(None)

            # Log extraction summary
            valid_count = sum(1 for e in embeddings if e is not None)
            if self._frame_count <= 5 or self._frame_count % 100 == 0:
                logger.info(
                    f"Frame {self._frame_count}: Extracted {valid_count}/{len(embeddings)} embeddings"
                )

            return embeddings

        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return [None] * len(frame_detections.detections)

    def _compute_embedding_similarity(
        self, emb1: np.ndarray | None, emb2: np.ndarray | None
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in range [-1, 1], or 0 if either is None.
        """
        if emb1 is None or emb2 is None:
            return 0.0

        # Both embeddings are already L2-normalized, so dot product = cosine similarity
        return float(np.dot(emb1, emb2))

    def _compute_color_similarity(
        self, hist1: np.ndarray | None, hist2: np.ndarray | None
    ) -> float:
        """Compute color histogram similarity using histogram intersection.

        Args:
            hist1: First color histogram.
            hist2: Second color histogram.

        Returns:
            Similarity score in range [0, 1], or 0 if either is None.
        """
        if hist1 is None or hist2 is None:
            return 0.0

        # Histogram intersection
        if hist1.shape != hist2.shape:
            return 0.0

        intersection = np.minimum(hist1, hist2).sum()
        total = min(hist1.sum(), hist2.sum())

        return float(intersection / total) if total > 0 else 0.0

    def _match_and_update_tracks(
        self,
        frame_detections: FrameDetections,
        detection_masks: list[np.ndarray | None],
        detection_embeddings: list[np.ndarray | None],
    ) -> FrameDetections:
        """Match detections to existing tracks using embeddings, IOU, and color.

        Matching priority:
        1. Embedding similarity (if re-identification enabled)
        2. Color histogram tiebreaker (when embedding scores are close)
        3. IOU matching (fallback, and for recently seen tracks)

        Args:
            frame_detections: New detections.
            detection_masks: Corresponding SAM2 masks.
            detection_embeddings: Feature embeddings for each detection.

        Returns:
            FrameDetections with tracking IDs assigned.
        """
        tracked_detections = []
        used_track_ids = set()

        for i, det in enumerate(frame_detections.detections):
            det_mask = detection_masks[i] if i < len(detection_masks) else None
            det_embedding = detection_embeddings[i] if i < len(detection_embeddings) else None
            det_bbox = det.bbox
            det_color = det.color_hist

            best_track_id = None
            best_score = 0.0
            second_best_score = 0.0
            second_best_track_id = None

            # Compare against existing tracks
            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue

                # Compute combined score
                score = 0.0
                frames_since_seen = self._frame_count - track.last_seen_frame

                # Primary: Embedding similarity (if available and track has embedding)
                if (
                    self._config.reidentification_enabled
                    and det_embedding is not None
                    and track.embedding is not None
                ):
                    emb_sim = self._compute_embedding_similarity(det_embedding, track.embedding)
                    # Weight embedding higher for recently seen tracks
                    if frames_since_seen <= 5:
                        score = emb_sim * 0.7  # Blend with IOU
                    else:
                        score = emb_sim  # Pure embedding for re-identification

                # Fallback/blend: IOU matching (for spatial consistency)
                if det_mask is not None and track.last_mask is not None:
                    iou = self._compute_mask_iou(det_mask, track.last_mask)
                elif track.last_bbox is not None:
                    iou = self._compute_bbox_iou(det_bbox, track.last_bbox)
                else:
                    iou = 0.0

                # For recently seen tracks, blend IOU with embedding
                if frames_since_seen <= 5:
                    if score > 0:
                        score = score + iou * 0.3  # Blend
                    else:
                        score = iou  # Pure IOU fallback

                # Track top two scores for tiebreaker
                if score > best_score:
                    second_best_score = best_score
                    second_best_track_id = best_track_id
                    best_score = score
                    best_track_id = track_id
                elif score > second_best_score:
                    second_best_score = score
                    second_best_track_id = track_id

            # Apply color tiebreaker if scores are close
            if (
                best_track_id is not None
                and second_best_track_id is not None
                and best_score - second_best_score < self._config.color_tiebreaker_threshold
                and det_color is not None
            ):
                best_track = self._tracks[best_track_id]
                second_track = self._tracks[second_best_track_id]

                color_sim_best = self._compute_color_similarity(det_color, best_track.color_hist)
                color_sim_second = self._compute_color_similarity(det_color, second_track.color_hist)

                if color_sim_second > color_sim_best:
                    # Swap - second track is better color match
                    best_track_id, second_best_track_id = second_best_track_id, best_track_id
                    best_score, second_best_score = second_best_score, best_score
                    logger.debug(
                        f"Color tiebreaker swapped track {second_best_track_id} -> {best_track_id}"
                    )

            # Decide: match existing track or create new
            # Use embedding threshold for re-ID, IOU threshold for spatial matching
            threshold = (
                self._config.embedding_similarity_threshold
                if self._config.reidentification_enabled and det_embedding is not None
                else self._config.min_iou_threshold
            )

            # Debug logging for tracking decisions
            if best_track_id is not None and best_score > 0:
                logger.debug(
                    f"Frame {self._frame_count}: Detection {i} best match: "
                    f"track {best_track_id} score={best_score:.3f} (threshold={threshold:.3f}), "
                    f"has_embedding={det_embedding is not None}"
                )

            if best_score >= threshold and best_track_id is not None:
                track_id = best_track_id
                used_track_ids.add(track_id)
                # Update existing track
                track = self._tracks[track_id]
                track.last_mask = det_mask
                track.last_bbox = det_bbox
                track.last_seen_frame = self._frame_count
                track.hit_count += 1
                # Update embedding with EMA (exponential moving average)
                if det_embedding is not None:
                    if track.embedding is None:
                        track.embedding = det_embedding
                    else:
                        # Blend new embedding with history (favor recent)
                        alpha = 0.7
                        track.embedding = alpha * det_embedding + (1 - alpha) * track.embedding
                        # Re-normalize
                        norm = np.linalg.norm(track.embedding)
                        if norm > 0:
                            track.embedding = track.embedding / norm
                # Update color histogram
                if det_color is not None:
                    track.color_hist = det_color
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
                    embedding=det_embedding,
                    color_hist=det_color,
                )
                logger.debug(
                    f"Frame {self._frame_count}: Created new track {track_id} "
                    f"(best_score={best_score:.3f}, threshold={threshold:.3f}, "
                    f"has_embedding={det_embedding is not None})"
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
        """Remove tracks not seen for too many frames.

        If lost_track_frames is 0, tracks are kept for the entire video.
        """
        if self._config.lost_track_frames <= 0:
            return  # Keep all tracks for entire video

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
