# SAM3 Tracking Integration Design

**Date:** 2026-01-14
**Status:** Approved
**Branch:** `feature/sam3-tracking-exploration`

## Problem Statement

The current tracking pipeline is broken:
- Track IDs change frequently (players get 50+ IDs instead of 1)
- Complex architecture: RF-DETR + SAM2 ImagePredictor + custom embedding extraction + custom matching logic
- SAM2 embedding extraction is fragile (relies on internal `_features` dict)
- No true temporal memory — frame-by-frame processing loses identity through occlusions

**Video characteristics:**
- Mobile phone, handheld/panning (camera follows action)
- Youth basketball with visually distinct players (team colors, undershirts, skin/hair, shoes)
- Batch processing acceptable (not real-time)

## Solution: SAM3 Text-Prompted Tracking

Replace the entire detection + tracking stack with SAM3's unified model.

### Architecture Comparison

**Current (broken):**
```
Video → RF-DETR → SAM2 ImagePredictor → Custom Embedding Extraction →
Custom Matching Logic → Track IDs → Jersey OCR → Player IDs
```
*6 components, fragile embedding extraction, no temporal memory*

**Proposed SAM3 Pipeline:**
```
Video → SAM3 VideoPredictor ("basketball player") → Stable Track IDs →
Jersey OCR → Player IDs
```
*3 components, built-in tracking, text-prompted detection*

### Why SAM3

| Feature | Benefit |
|---------|---------|
| Text prompts | "basketball player" finds all players — no RF-DETR needed |
| Built-in memory bank | Maintains identity across frames (inherited from SAM2, improved) |
| Separate detector/tracker | Reduces ID-swap problem |
| Presence token | Better discrimination between similar prompts ("player in white" vs "player in dark") |
| Single 848M param model | Replaces RF-DETR + SAM2 + custom logic |

## Hardware Strategy

**Primary:** Apple Silicon Mac (MPS)
- SAM3 works on MPS via HuggingFace transformers implementation
- Requires patch for `pin_memory()` in video processing

**Fallback:** Windows RTX 3090 (CUDA)
- Full CUDA 12.6+ support
- Use if MPS proves unstable

**Installation:**
```bash
# HuggingFace transformers (MPS compatible, avoids Triton dependency)
pip install git+https://github.com/huggingface/transformers torchvision
```

## Video Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PREPARATION                                                  │
│    Video → Extract to JPEG folder (SAM3 requirement)            │
│    └── /tmp/sam3_frames/{video_id}/000000.jpg, 000001.jpg, ...  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. SESSION START                                                │
│    predictor.handle_request(type="start_session",               │
│                             resource_path=jpeg_folder)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ADD TEXT PROMPT (frame 0)                                    │
│    predictor.handle_request(type="add_prompt",                  │
│                             frame_index=0,                      │
│                             text="basketball player")           │
│    → Returns initial detections + object IDs                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. PROPAGATE THROUGH VIDEO                                      │
│    for result in predictor.propagate_in_video(direction="fwd"): │
│        frame_idx = result["frame_index"]                        │
│        masks = result["masks"]        # Per-object masks        │
│        boxes = result["boxes"]        # Per-object bboxes       │
│        obj_ids = result["object_ids"] # Stable across frames    │
│        → Convert to FrameDetections                             │
│        → Store in database                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. JERSEY OCR (parallel, on detected crops)                     │
│    For each track, sample frames with good visibility           │
│    → SmolVLM2 reads jersey numbers                              │
│    → Aggregate results per track_id                             │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### New Files

```
backend/app/ml/
├── sam3_tracker.py          # Main SAM3 VideoPredictor wrapper
├── sam3_frame_extractor.py  # Video → JPEG folder extraction

backend/app/services/
├── sam3_detection_pipeline.py  # Dedicated SAM3 pipeline (Option A)
```

### Modified Files

```
backend/app/config.py        # Add sam3_* settings
backend/requirements.txt     # Add transformers from git
```

### Core Class Design

```python
class SAM3VideoTracker:
    """SAM3-based tracker using text-prompted video segmentation."""

    def __init__(self, config: SAM3TrackerConfig):
        self.predictor = None  # Lazy-loaded
        self.session_id = None

    def process_video(
        self,
        video_path: Path,
        prompt: str = "basketball player",
        sample_interval: int = 3,
    ) -> Generator[FrameDetections, None, None]:
        """Process entire video, yielding FrameDetections per frame."""

    def _convert_to_frame_detections(
        self,
        sam3_output: dict,
        frame_number: int,
    ) -> FrameDetections:
        """Convert SAM3 output to existing FrameDetections type."""
```

## Configuration

```python
# backend/app/config.py additions

# SAM3 settings
sam3_model_path: Path = models_dir / "sam3.pt"
sam3_prompt: str = "basketball player"
sam3_confidence_threshold: float = 0.25
sam3_use_half_precision: bool = True  # FP16 for speed
sam3_temp_frames_dir: Path = Path("/tmp/sam3_frames")
```

## Error Handling

| Error Type | Handling |
|------------|----------|
| Model download fails | Retry with exponential backoff, clear error message |
| HuggingFace auth missing | Error: "Run `huggingface-cli login` first" |
| Out of memory (OOM) | Fall back to CPU, then fail gracefully |
| MPS operation unsupported | Automatic fallback to CPU with warning |
| Video corrupt/unreadable | Skip video, log error, continue |
| No players detected | Return empty FrameDetections, log warning |

### Device Selection

```python
def _select_device(self) -> str:
    """Priority: CUDA > MPS > CPU with validation."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Test MPS actually works
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"
```

## Jersey OCR Integration

SAM3 provides segmentation masks — use for better OCR:

```python
def extract_jersey_crop(frame, mask, bbox) -> np.ndarray:
    """Extract jersey region using SAM3 mask.

    1. Crop to upper 40% of bbox (jersey area)
    2. Apply mask to remove background
    """
    x1, y1, x2, y2 = bbox.to_xyxy()
    jersey_y2 = y1 + int((y2 - y1) * 0.4)

    crop = frame[y1:jersey_y2, x1:x2]
    crop_mask = mask[y1:jersey_y2, x1:x2]
    crop[~crop_mask] = 0  # Zero background
    return crop
```

**OCR aggregation:** Majority vote across sampled frames per track.

## Output Format

Uses existing types for compatibility:

```python
@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int              # 0 = person
    class_name: str            # "person"
    tracking_id: int           # SAM3's stable object ID
    mask: np.ndarray | None    # NEW: SAM3 segmentation mask
    jersey_number: str | None  # Populated after OCR
```

Maps directly to existing `player_detections` table.

## Implementation Phases

### Phase 1: SAM3 Core Integration (MVP)

| Step | Task |
|------|------|
| 1.1 | Install SAM3 via HuggingFace transformers |
| 1.2 | Create `sam3_tracker.py` with `SAM3VideoTracker` |
| 1.3 | Create `sam3_frame_extractor.py` for JPEG extraction |
| 1.4 | Add SAM3 config settings to `config.py` |
| 1.5 | Test on single video with "basketball player" prompt |

### Phase 2: Pipeline Integration

| Step | Task |
|------|------|
| 2.1 | Create `sam3_detection_pipeline.py` |
| 2.2 | Integrate with existing `FrameDetections` types |
| 2.3 | Wire up database storage |
| 2.4 | Add `tracking_backend: "sam3"` config option |

### Phase 3: Jersey OCR Integration

| Step | Task |
|------|------|
| 3.1 | Implement mask-based jersey crop extraction |
| 3.2 | Integrate existing SmolVLM2 OCR |
| 3.3 | Aggregate OCR results per track |

### Phase 4: Polish & Cleanup

| Step | Task |
|------|------|
| 4.1 | Add progress reporting / logging |
| 4.2 | Test MPS on Mac |
| 4.3 | Test CUDA on RTX 3090 (fallback) |
| 4.4 | Remove or deprecate old SAM2 tracker |

## Success Criteria

1. **Track stability:** Same player keeps same ID for entire video
2. **No fragmentation:** ~10 track IDs per video (not 50+)
3. **Jersey OCR works:** Can identify specific players by number
4. **Performance:** < 5 min for 10 min video on MPS

## References

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 HuggingFace](https://huggingface.co/facebook/sam3)
- [SAM3 MPS Workaround](https://huggingface.co/facebook/sam3/discussions/11)
- [Ultralytics SAM3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [Basketball Player Tracking Best Practices](https://blog.roboflow.com/identify-basketball-players/)
