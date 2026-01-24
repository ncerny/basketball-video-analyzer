# Tracking Implementation Options Spec

This document outlines three alternative tracking approaches to replace/improve the current SAM2 ImagePredictor-based tracking.

## Current Problem

The current SAM2 tracker uses ImagePredictor (per-frame segmentation) with custom tracking logic. This is fundamentally flawed because:
1. SAM2 embeddings are optimized for segmentation, not re-identification
2. ImagePredictor has no temporal context
3. Our custom matching logic fights against SAM2's design

---

## Option A: Hybrid SAM2 + Norfair

### Overview
Use SAM2 **only for mask extraction** (its strength) and Norfair for actual tracking logic. Norfair is a proven tracking library with built-in Kalman filter, ReID support, and optimized association.

### Rationale
- Leverages existing code (both SAM2 and Norfair are already integrated)
- SAM2 masks improve color histogram quality (exclude background)
- Norfair handles temporal association with battle-tested algorithms
- Minimal new dependencies

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         Per Frame                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   RF-DETR    │───>│    SAM2      │───>│  Color Extract   │  │
│  │  Detection   │    │ Mask Only    │    │  (with masks)    │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│         │                                          │            │
│         │ bboxes                                   │ color_hist │
│         ▼                                          ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Norfair Tracker                          ││
│  │  - Kalman filter for motion prediction                      ││
│  │  - Euclidean/IOU distance for association                   ││
│  │  - Color histogram ReID for occlusion recovery              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│                      tracked_detections                         │
└─────────────────────────────────────────────────────────────────┘
```

### Dependencies
- `norfair` (already installed)
- `sam2` (already installed)
- No new dependencies required

### Implementation Steps

#### Step 1: Create SAM2MaskExtractor (new file)
```python
# app/ml/sam2_mask_extractor.py
"""SAM2 mask extraction only - no tracking logic."""

class SAM2MaskExtractor:
    """Extract precise segmentation masks using SAM2 ImagePredictor."""

    def __init__(self, model_name: str = "sam2_hiera_tiny", device: str = "auto"):
        self._predictor = None  # Lazy load

    def extract_masks(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> list[np.ndarray | None]:
        """Generate SAM2 masks for all detections.

        Args:
            frame: BGR image
            detections: List of Detection objects with bboxes

        Returns:
            List of binary masks (same order as detections)
        """
        # Implementation: batch predict masks from bboxes
        pass
```

#### Step 2: Enhance NorfairTracker to accept masks
```python
# Modify app/ml/norfair_tracker.py

class NorfairTracker:
    def update(
        self,
        frame_detections: FrameDetections,
        frame: np.ndarray | None = None,  # NEW: for mask-based color extraction
        masks: list[np.ndarray | None] | None = None,  # NEW: SAM2 masks
    ) -> FrameDetections:
        # If masks provided, extract colors using masks
        # Otherwise fall back to bbox-only extraction
        pass
```

#### Step 3: Update detection pipeline
```python
# Modify app/services/detection_pipeline.py

# When tracking_backend == "norfair":
# 1. Run RF-DETR detection
# 2. Run SAM2 mask extraction
# 3. Extract colors using masks
# 4. Pass to Norfair with enhanced color features
```

#### Step 4: Tune Norfair parameters
```python
# Recommended starting parameters for basketball:
NorfairTracker(
    distance_function="iou",  # Use IOU instead of euclidean
    distance_threshold=0.5,   # 50% IOU minimum
    hit_counter_max=90,       # 3 seconds at 30fps
    initialization_delay=2,   # Require 2 frames to confirm
    reid_distance_threshold=0.4,  # Stricter color matching
    reid_hit_counter_max=150,     # 5 seconds for ReID
)
```

### Files to Modify
| File | Changes |
|------|---------|
| `app/ml/sam2_mask_extractor.py` | NEW: Mask-only extraction |
| `app/ml/norfair_tracker.py` | Add mask support, tune params |
| `app/services/detection_pipeline.py` | Integrate mask extraction |
| `app/config.py` | Add norfair tuning params |

### Configuration Parameters
```python
# config.py additions
norfair_distance_function: str = "iou"
norfair_distance_threshold: float = 0.5
norfair_hit_counter_max: int = 90
norfair_initialization_delay: int = 2
norfair_reid_threshold: float = 0.4
norfair_reid_hit_counter_max: int = 150
norfair_use_sam2_masks: bool = True  # Toggle mask extraction
```

### Expected Benefits
- Stable frame-to-frame tracking (Norfair's Kalman filter)
- Better color features (SAM2 masks exclude background)
- Proven ReID for occlusion recovery
- No new dependencies

### Limitations
- Still relies on color histograms for ReID (not deep features)
- Two-stage inference (detector + SAM2) adds latency
- May still struggle with very similar jerseys

### Complexity: LOW-MEDIUM
- Mostly refactoring existing code
- ~200-300 lines of new code

---

## Option B: BoT-SORT/StrongSORT via BoxMOT

### Overview
Replace custom tracking with BoxMOT's production-ready trackers. BoxMOT provides pluggable SOTA trackers (BoT-SORT, StrongSORT, DeepOCSORT) that work with any detector outputting bboxes.

### Rationale
- State-of-the-art tracking algorithms
- Deep ReID features (not just color histograms)
- Camera motion compensation (GMC)
- Battle-tested on MOT benchmarks
- Clean API for custom detectors

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         Per Frame                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌─────────────────────────────────┐  │
│  │   RF-DETR    │────────>│         BoxMOT Tracker          │  │
│  │  Detection   │ N×6     │  (BoT-SORT / StrongSORT)        │  │
│  └──────────────┘ array   │                                 │  │
│                           │  ┌─────────┐  ┌──────────────┐  │  │
│  ┌──────────────┐         │  │ Kalman  │  │  ReID Model  │  │  │
│  │    Frame     │────────>│  │ Filter  │  │ (OSNet etc)  │  │  │
│  │   (BGR)      │         │  └─────────┘  └──────────────┘  │  │
│  └──────────────┘         │                                 │  │
│                           │  ┌─────────────────────────────┐│  │
│                           │  │ Camera Motion Compensation  ││  │
│                           │  │ (ECC / SparseOptFlow)       ││  │
│                           │  └─────────────────────────────┘│  │
│                           └────────────────┬────────────────┘  │
│                                            │                   │
│                                            ▼                   │
│                                   tracked_detections           │
│                                   N×8 (x,y,x,y,id,conf,cls,idx)│
└─────────────────────────────────────────────────────────────────┘
```

### Dependencies
```
boxmot>=10.0.0
# Includes: BoT-SORT, StrongSORT, DeepOCSORT, ByteTrack, OC-SORT
# ReID models: OSNet, CLIP, etc.
```

### Implementation Steps

#### Step 1: Install BoxMOT
```bash
pip install boxmot
# Or add to pyproject.toml
```

#### Step 2: Create BoxMOT tracker wrapper
```python
# app/ml/boxmot_tracker.py
"""BoxMOT tracker integration for SOTA multi-object tracking."""

from pathlib import Path
import numpy as np
from boxmot import StrongSORT, BoTSORT, DeepOCSORT

from .types import Detection, FrameDetections, BoundingBox


class BoxMOTTracker:
    """Wrapper for BoxMOT trackers (StrongSORT, BoT-SORT, etc.)."""

    TRACKER_CLASSES = {
        "strongsort": StrongSORT,
        "botsort": BoTSORT,
        "deepocsort": DeepOCSORT,
    }

    def __init__(
        self,
        tracker_type: str = "strongsort",
        reid_model: str = "osnet_x0_25_msmt17.pt",
        device: str = "cuda:0",
        half: bool = False,
    ):
        self._tracker_type = tracker_type
        tracker_cls = self.TRACKER_CLASSES[tracker_type]

        self._tracker = tracker_cls(
            model_weights=Path(reid_model),
            device=device,
            fp16=half,
        )

    def update(
        self,
        frame_detections: FrameDetections,
        frame: np.ndarray,
    ) -> FrameDetections:
        """Update tracker with new detections.

        Args:
            frame_detections: Detections from RF-DETR
            frame: BGR frame image (required for ReID)

        Returns:
            FrameDetections with tracking IDs assigned
        """
        # Convert to BoxMOT format: N x (x1, y1, x2, y2, conf, cls)
        dets = self._to_boxmot_format(frame_detections)

        # Run tracker
        tracks = self._tracker.update(dets, frame)
        # Returns: M x (x1, y1, x2, y2, id, conf, cls, idx)

        return self._from_boxmot_format(tracks, frame_detections)

    def _to_boxmot_format(self, frame_detections: FrameDetections) -> np.ndarray:
        """Convert FrameDetections to BoxMOT format."""
        dets = []
        for det in frame_detections.detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            dets.append([x1, y1, x2, y2, det.confidence, det.class_id])
        return np.array(dets) if dets else np.empty((0, 6))

    def _from_boxmot_format(
        self,
        tracks: np.ndarray,
        original: FrameDetections
    ) -> FrameDetections:
        """Convert BoxMOT tracks back to FrameDetections."""
        detections = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, idx = track
            detections.append(Detection(
                bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=int(cls),
                class_name="person",
                tracking_id=int(track_id),
            ))
        return FrameDetections(
            frame_number=original.frame_number,
            detections=detections,
            frame_width=original.frame_width,
            frame_height=original.frame_height,
        )

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.reset()
```

#### Step 3: Update detection pipeline
```python
# Modify app/services/detection_pipeline.py

from app.ml.boxmot_tracker import BoxMOTTracker

def _get_tracker(self, fps: float):
    if settings.tracking_backend == "strongsort":
        return BoxMOTTracker(
            tracker_type="strongsort",
            reid_model=settings.reid_model_path,
            device=self._config.device,
        )
    elif settings.tracking_backend == "botsort":
        return BoxMOTTracker(
            tracker_type="botsort",
            reid_model=settings.reid_model_path,
            device=self._config.device,
        )
```

#### Step 4: Download ReID model
```bash
# OSNet is lightweight and effective
wget https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.0/osnet_x0_25_msmt17.pt \
    -O models/osnet_x0_25_msmt17.pt
```

### Files to Modify/Create
| File | Changes |
|------|---------|
| `app/ml/boxmot_tracker.py` | NEW: BoxMOT wrapper |
| `app/services/detection_pipeline.py` | Add BoxMOT tracker option |
| `app/config.py` | Add tracking_backend options, reid_model_path |
| `pyproject.toml` | Add boxmot dependency |

### Configuration Parameters
```python
# config.py additions
tracking_backend: Literal["bytetrack", "norfair", "sam2", "strongsort", "botsort"] = "strongsort"
reid_model_path: str = "./models/osnet_x0_25_msmt17.pt"
reid_model_device: str = "auto"

# StrongSORT specific
strongsort_max_dist: float = 0.2
strongsort_max_iou_dist: float = 0.7
strongsort_max_age: int = 70
strongsort_n_init: int = 3

# BoT-SORT specific
botsort_proximity_thresh: float = 0.5
botsort_appearance_thresh: float = 0.25
botsort_gmc_method: str = "sparseOptFlow"
```

### Expected Benefits
- Deep ReID features (much better than color histograms)
- Camera motion compensation
- State-of-the-art identity preservation
- Handles occlusions well
- Widely used in production

### Limitations
- Additional dependency (boxmot)
- Requires ReID model download (~25MB for OSNet)
- Slightly higher latency due to ReID inference
- May need GPU for real-time performance

### Complexity: MEDIUM
- Clean integration via wrapper class
- ~150-200 lines of new code
- Need to download ReID model weights

---

## Option E: SAM2MOT

### Overview
Implement the SAM2MOT framework which achieves state-of-the-art results on DanceTrack (+2.1 HOTA, +4.5 IDF1). It uses SAM2 VideoPredictor with two key innovations:
1. **Trajectory Manager System** - handles object addition/removal
2. **Cross-Object Interaction** - resolves occlusions

### Rationale
- State-of-the-art on DanceTrack (similar motion to basketball)
- Zero-shot generalization (no training required)
- Leverages SAM2's temporal memory
- Designed specifically for SAM2 + MOT

### Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SAM2MOT Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Trajectory Manager System                       │ │
│  │                                                                    │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐│ │
│  │  │   RF-DETR    │───>│  Hungarian   │───>│   Overlap Check      ││ │
│  │  │  Detection   │    │  Matching    │    │ (untracked regions)  ││ │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘│ │
│  │         │                   │                      │              │ │
│  │         ▼                   ▼                      ▼              │ │
│  │   High-conf dets     Matched tracks         New objects          │ │
│  │                                                                    │ │
│  └────────────────────────────┬──────────────────────────────────────┘ │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    SAM2 VideoPredictor                            │ │
│  │                                                                    │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐│ │
│  │  │   Memory     │───>│    Mask      │───>│    Propagate        ││ │
│  │  │   Bank       │    │  Prediction  │    │    to Frame          ││ │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘│ │
│  │                                                                    │ │
│  └────────────────────────────┬──────────────────────────────────────┘ │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                Cross-Object Interaction Module                     │ │
│  │                                                                    │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐│ │
│  │  │  Mask IOU    │───>│   Occlusion  │───>│   Resolve via       ││ │
│  │  │  Detection   │    │   Detection  │    │   Logits Variance    ││ │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘│ │
│  │                                                                    │ │
│  └────────────────────────────┬──────────────────────────────────────┘ │
│                               │                                         │
│                               ▼                                         │
│                       tracked_detections                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependencies
```
sam2>=1.0  # Already installed
mmdet>=3.3.0  # Optional: for Co-DINO detector
scipy  # For Hungarian matching (already installed)
```

### Implementation Steps

#### Step 1: Create SAM2MOT Tracker
```python
# app/ml/sam2mot_tracker.py
"""SAM2MOT: Multi-Object Tracking by Segmentation."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import settings
from .types import Detection, FrameDetections, BoundingBox


class TrajectoryManager:
    """Manages object lifecycle: addition, removal, quality reconstruction."""

    def __init__(
        self,
        det_conf_threshold: float = 0.5,
        new_object_iou_threshold: float = 0.3,
    ):
        self._det_conf_threshold = det_conf_threshold
        self._new_object_iou_threshold = new_object_iou_threshold
        self._active_tracks: dict[int, dict] = {}
        self._untracked_mask: np.ndarray | None = None

    def process_detections(
        self,
        detections: list[Detection],
        current_masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> tuple[list[int], list[Detection]]:
        """Three-stage filtering for detection processing.

        Stage 1: High-confidence detection selection
        Stage 2: Hungarian matching to existing tracks
        Stage 3: Overlap check with untracked regions

        Returns:
            matched_track_ids: Track IDs that matched detections
            new_objects: Detections that should become new tracks
        """
        # Stage 1: Filter high-confidence detections
        high_conf_dets = [d for d in detections if d.confidence >= self._det_conf_threshold]

        # Stage 2: Hungarian matching
        matched_track_ids, unmatched_dets = self._hungarian_match(
            high_conf_dets, current_masks
        )

        # Stage 3: Check unmatched against untracked region mask
        new_objects = self._filter_new_objects(unmatched_dets, current_masks, frame_shape)

        return matched_track_ids, new_objects

    def _hungarian_match(
        self,
        detections: list[Detection],
        current_masks: dict[int, np.ndarray],
    ) -> tuple[list[int], list[Detection]]:
        """Match detections to existing tracks using Hungarian algorithm."""
        # Build cost matrix using bbox IOU
        # Return matched track IDs and unmatched detections
        pass

    def _filter_new_objects(
        self,
        detections: list[Detection],
        current_masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        """Filter detections that don't overlap with tracked objects."""
        # Compute untracked region mask
        # Check overlap with each detection
        pass


class CrossObjectInteraction:
    """Resolves occlusion-related tracking errors."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        logits_history_frames: int = 10,
    ):
        self._iou_threshold = iou_threshold
        self._logits_history: dict[int, list[float]] = {}
        self._history_frames = logits_history_frames

    def detect_and_resolve_occlusions(
        self,
        masks: dict[int, np.ndarray],
        logits: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Detect occlusions via mask IOU and resolve via logits analysis.

        Two-stage approach:
        1. Compare logits scores of overlapping objects
        2. If similar, use variance over past N frames to identify occluded object

        Returns:
            Corrected masks with occlusion handling
        """
        # Compute pairwise mask IOUs
        overlapping_pairs = self._find_overlapping_pairs(masks)

        for track_id_a, track_id_b in overlapping_pairs:
            # Identify which track is occluded
            occluded_id = self._identify_occluded(track_id_a, track_id_b, logits)
            # Apply correction to occluded track's mask
            if occluded_id:
                masks = self._correct_occlusion(masks, track_id_a, track_id_b, occluded_id)

        return masks


class SAM2MOTTracker:
    """SAM2MOT: Multi-Object Tracking using SAM2 VideoPredictor."""

    def __init__(
        self,
        model_name: str = "sam2_hiera_tiny",
        device: str = "auto",
        det_conf_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._device = device

        # Lazy-loaded components
        self._video_predictor = None
        self._inference_state = None

        # Sub-modules
        self._trajectory_manager = TrajectoryManager(det_conf_threshold=det_conf_threshold)
        self._cross_object = CrossObjectInteraction()

        # Track management
        self._next_track_id = 1
        self._frame_idx = 0

    def _load_model(self) -> None:
        """Load SAM2 VideoPredictor."""
        if self._video_predictor is not None:
            return

        from sam2.build_sam import build_sam2_video_predictor

        # Build video predictor (not image predictor!)
        self._video_predictor = build_sam2_video_predictor(
            config_file=f"configs/sam2.1/sam2.1_hiera_{self._model_name.split('_')[-1]}.yaml",
            ckpt_path=str(settings.models_dir / f"{self._model_name}.pt"),
            device=self._device,
        )

    def init_video(self, video_path: str) -> None:
        """Initialize video predictor with video file."""
        self._load_model()
        self._inference_state = self._video_predictor.init_state(video_path=video_path)
        self._frame_idx = 0
        self._trajectory_manager = TrajectoryManager()

    def update(
        self,
        frame_detections: FrameDetections,
        frame: np.ndarray,
    ) -> FrameDetections:
        """Process frame with SAM2MOT pipeline."""

        # Step 1: Propagate existing tracks to current frame
        current_masks, logits = self._propagate_masks()

        # Step 2: Trajectory Manager - handle new/removed objects
        matched_ids, new_objects = self._trajectory_manager.process_detections(
            frame_detections.detections,
            current_masks,
            (frame_detections.frame_height, frame_detections.frame_width),
        )

        # Step 3: Add new objects to SAM2
        for det in new_objects:
            track_id = self._add_new_track(det, frame)
            matched_ids.append(track_id)

        # Step 4: Cross-Object Interaction - resolve occlusions
        corrected_masks = self._cross_object.detect_and_resolve_occlusions(
            current_masks, logits
        )

        # Step 5: Convert masks to detections
        tracked_detections = self._masks_to_detections(
            corrected_masks, frame_detections
        )

        self._frame_idx += 1

        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=tracked_detections,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
        )

    def _propagate_masks(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Propagate all tracked objects to current frame."""
        # Use SAM2 VideoPredictor propagate_in_video
        pass

    def _add_new_track(self, detection: Detection, frame: np.ndarray) -> int:
        """Add new track to SAM2 with bbox prompt."""
        track_id = self._next_track_id
        self._next_track_id += 1

        # Add to SAM2 using add_new_points_or_box
        x1, y1, x2, y2 = detection.bbox.to_xyxy()
        self._video_predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=self._frame_idx,
            obj_id=track_id,
            box=np.array([x1, y1, x2, y2]),
        )

        return track_id
```

#### Step 2: Integrate with detection pipeline
```python
# Modify app/services/detection_pipeline.py

# SAM2MOT requires video-level processing, not frame-by-frame
# This requires restructuring the pipeline to:
# 1. Initialize SAM2MOT with full video
# 2. Process frames in order
# 3. Handle streaming scenario differently
```

#### Step 3: Handle video vs streaming
```python
# SAM2MOT works best with video files (uses VideoPredictor)
# For streaming/API use, need adaptation:

class SAM2MOTStreamingAdapter:
    """Adapts SAM2MOT for frame-by-frame streaming."""

    def __init__(self, buffer_size: int = 30):
        self._buffer: list[np.ndarray] = []
        self._buffer_size = buffer_size
        # Use ImagePredictor with custom memory management
```

### Files to Modify/Create
| File | Changes |
|------|---------|
| `app/ml/sam2mot_tracker.py` | NEW: Full SAM2MOT implementation |
| `app/ml/sam2mot_trajectory.py` | NEW: Trajectory Manager |
| `app/ml/sam2mot_interaction.py` | NEW: Cross-Object Interaction |
| `app/services/detection_pipeline.py` | Add SAM2MOT option |
| `app/services/batch_processor.py` | Video-level processing support |
| `app/config.py` | SAM2MOT parameters |

### Configuration Parameters
```python
# config.py additions
tracking_backend: Literal[...,"sam2mot"] = "sam2mot"

# SAM2MOT specific
sam2mot_det_conf_threshold: float = 0.5
sam2mot_new_object_iou_threshold: float = 0.3
sam2mot_occlusion_iou_threshold: float = 0.3
sam2mot_logits_history_frames: int = 10
sam2mot_streaming_buffer_size: int = 30
```

### Expected Benefits
- State-of-the-art tracking accuracy (DanceTrack: 75.8 HOTA)
- Zero-shot generalization
- Handles occlusions via cross-object interaction
- Leverages SAM2's temporal memory
- Precise mask-level tracking

### Limitations
- More complex implementation
- Requires VideoPredictor (video-file oriented)
- Streaming adaptation needed for API use
- Higher computational requirements
- Research code, not production-tested

### Complexity: HIGH
- Significant new code (~500-800 lines)
- VideoPredictor vs ImagePredictor migration
- Streaming adaptation required
- Less mature than Option B

---

## Comparison Summary

| Aspect | Option A (SAM2+Norfair) | Option B (BoxMOT) | Option E (SAM2MOT) |
|--------|------------------------|-------------------|-------------------|
| **Complexity** | LOW-MEDIUM | MEDIUM | HIGH |
| **New Code** | ~200-300 lines | ~150-200 lines | ~500-800 lines |
| **Dependencies** | None new | boxmot (~25MB) | None new |
| **ReID Quality** | Color histograms | Deep features | SAM2 memory |
| **Maturity** | Proven (Norfair) | Proven (BoxMOT) | Research |
| **SOTA Performance** | Good | Very Good | Best (on DanceTrack) |
| **Streaming Ready** | Yes | Yes | Needs adaptation |
| **GPU Memory** | Low | Medium | High |

## Recommended Implementation Order

1. **Option A first** - Quick win, minimal risk, validates mask-based color extraction
2. **Option B second** - Production-ready SOTA tracking with deep ReID
3. **Option E third** - Research implementation for maximum accuracy

---

## References

- [Norfair GitHub](https://github.com/tryolabs/norfair)
- [BoxMOT GitHub](https://github.com/mikel-brostrom/boxmot)
- [SAM2MOT Paper](https://arxiv.org/html/2504.04519v1)
- [SAM2MOT GitHub](https://github.com/TripleJoy/SAM2MOT)
- [Ultralytics Tracking Docs](https://docs.ultralytics.com/modes/track/)
