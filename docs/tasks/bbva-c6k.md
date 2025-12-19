# bbva-c6k: Fix Tracking Stability - ByteTrack Integration Bugs

**Status**: Complete (Pending Manual Validation)  
**Priority**: P0  
**Type**: Bug  
**Created**: 2025-12-19  
**Completed**: 2025-12-19  

## Problem Statement

Detection and tracking "skips in and out inconsistently" - players appear/disappear between frames and tracking IDs are not stable.

## Root Cause Analysis

Deep analysis identified **6 root causes** (3 critical bugs, 3 configuration issues):

### 🔴 Bug 1: Incorrect ByteTrack ID Mapping (CRITICAL)

**Location**: `backend/app/ml/byte_tracker.py` lines 92-124

**Problem**: The `_update_tracking_ids()` method assumes index alignment between original detections and tracked outputs. ByteTrack **filters and reorders** detections, so:
- When counts match: IDs assigned by index (wrong order)
- When counts differ: IDs assigned to "first N" (completely wrong)

**Current Code**:
```python
if num_tracked == len(original.detections):
    for i, detection in enumerate(original.detections):
        detection.tracking_id = int(tracked.tracker_id[i])  # WRONG ORDER!
else:
    for i in range(min(num_tracked, len(original.detections))):
        original.detections[i].tracking_id = int(tracked.tracker_id[i])  # WRONG!
```

**Fix**: Convert FROM `sv.Detections` (use returned detections as source of truth), not map IDs back to original.

---

### 🔴 Bug 2: Filtering BEFORE Tracking (CRITICAL)

**Location**: `backend/app/services/detection_pipeline.py` lines 337-355

**Current Pipeline**:
```
YOLO Detection → Court Filter → ByteTrack Tracking
```

**Problem**: If court detection is inconsistent frame-to-frame, a player gets filtered in some frames but not others. ByteTrack never sees these detections, so it can't maintain tracking continuity.

**Fix**: Reorder to:
```
YOLO Detection → ByteTrack Tracking → Court Filter
```

---

### 🔴 Bug 3: Frame Sampling Breaks Tracker Timing (CRITICAL)

**Location**: `backend/app/services/detection_pipeline.py` line 295

**Problem**: 
- `sample_interval=3` means tracker sees ~10 FPS on 30 FPS video
- Tracker initialized with `frame_rate=metadata.fps` (30)
- Motion predictions and `lost_track_buffer` timing are **3x off**

**Fix**: Pass effective FPS:
```python
effective_fps = metadata.fps / self._config.sample_interval
tracker = self._get_tracker(effective_fps)
```

---

### 🟠 Bug 4: Empty Frames Don't Advance Tracker (HIGH)

**Location**: `backend/app/ml/byte_tracker.py` lines 49-50

**Problem**: Early return on empty detections skips tracker update:
```python
if not frame_detections.detections:
    return frame_detections  # Tracker state never advances!
```

ByteTrack needs updates even on empty frames to age lost tracks properly.

**Fix**: Always call `update_with_detections()`.

---

### 🟠 Bug 5: Court Detection Fallback Floods Tracker (HIGH)

**Location**: `backend/app/ml/court_detector.py` lines 75-77

**Problem**: When no court lines detected:
```python
return np.ones((height, width), dtype=np.uint8) * 255  # Full frame = "court"
```

This passes ALL detections (including audience) to tracking, causing ID churn.

**Fix**: Cache last good mask, use conservative central fallback.

---

### 🟡 Bug 6: Parameters Too Strict (MEDIUM)

| Parameter | Current | Recommended | Issue |
|-----------|---------|-------------|-------|
| `confidence_threshold` | 0.5 | 0.35 | Missing partially-occluded players |
| `tracking_iou_threshold` | 0.8 | 0.55-0.6 | Too strict for fast motion |
| `tracking_buffer_seconds` | 1.0 | 2.0-3.0 | Too short for occlusions |

---

## Implementation Plan

### Phase A: Critical Bug Fixes (Priority: Immediate)

1. **Fix ByteTrack ID mapping** (`byte_tracker.py`)
   - Replace `_update_tracking_ids()` with `_from_supervision_detections()`
   - Convert back from `sv.Detections` to our `FrameDetections` format
   - Use the tracked detections as source of truth

2. **Fix empty frame handling** (`byte_tracker.py`)
   - Always call `update_with_detections()` even with empty detections
   - Return empty `FrameDetections` after updating tracker state

3. **Fix tracker timing** (`detection_pipeline.py`)
   - Calculate `effective_fps = fps / sample_interval`
   - Pass effective FPS to tracker initialization
   - Scale `lost_track_buffer` accordingly

4. **Reorder pipeline** (`detection_pipeline.py`)
   - Move ByteTrack tracking BEFORE court filtering
   - Apply court filter to tracked detections (preserves tracking continuity)

### Phase B: Stability Improvements (Priority: High)

5. **Court mask caching** (`court_detector.py`)
   - Cache last successfully detected court mask
   - Use cached mask when detection fails
   - Conservative central fallback if no cached mask

6. **Filter persons only** (`detection_pipeline.py`)
   - Track only person detections (class_id=0)
   - Keep ball detections separate

7. **Tracker reset** (`detection_pipeline.py`)
   - Call `tracker.reset()` at start of each video

### Phase C: Parameter Tuning (Priority: Medium)

8. **Update defaults** (`config.py`, `detection_pipeline.py`)
   - Lower `confidence_threshold` to 0.35
   - Lower `tracking_iou_threshold` to 0.6
   - Increase `tracking_buffer_seconds` to 2.0

### Phase D: Validation (Priority: High)

9. **Run existing tests** - Verify fixes don't break anything
10. **Manual testing** - Validate tracking stability with real video

---

## Files to Modify

| File | Changes |
|------|---------|
| `backend/app/ml/byte_tracker.py` | Replace ID mapping, fix empty frame handling |
| `backend/app/services/detection_pipeline.py` | Reorder pipeline, fix timing, filter persons |
| `backend/app/ml/court_detector.py` | Add mask caching |
| `backend/app/config.py` | Tune default parameters |

---

## Acceptance Criteria

- [ ] Tracking IDs remain stable across frames (no random switches) - *Needs manual testing*
- [ ] Players don't "skip in and out" between frames - *Needs manual testing*
- [x] Existing tests pass - **274 tests pass**
- [x] Court filtering works without breaking tracking continuity - *Pipeline reordered*
- [x] Processing performance not significantly degraded - *No additional processing added*

---

## Implementation Details

### Completed: 2025-12-19

All fixes implemented and validated. **274 tests pass.**

---

### 1. ByteTrack ID Mapping Fix (`backend/app/ml/byte_tracker.py`)

**Changes Made:**

1. **Added `CLASS_NAMES` constant** for reverse class ID mapping:
   ```python
   CLASS_NAMES: dict[int, str] = {
       0: "person",
       32: "sports_ball",
   }
   ```

2. **Replaced `_update_tracking_ids()` with `_from_supervision_detections()`**:
   - New method properly converts ByteTrack's `sv.Detections` output back to our `FrameDetections` format
   - Creates new `Detection` objects from `tracked.xyxy`, `tracked.confidence`, `tracked.class_id`, `tracked.tracker_id`
   - Uses ByteTrack's output as the source of truth (not the original input detections)

3. **Fixed `update()` method**:
   - Removed early return for empty detections
   - Now always calls `update_with_detections()` even for empty input
   - Uses new `_from_supervision_detections()` for result conversion

---

### 2. Detection Pipeline Fixes (`backend/app/services/detection_pipeline.py`)

**Changes Made:**

1. **Fixed pipeline order** (Track BEFORE Court Filter):
   - Changed from: `YOLO detect → Court filter → ByteTrack track`
   - Changed to: `YOLO detect → ByteTrack track → Court filter`

2. **Fixed frame sampling timing**:
   - Now calculates `effective_fps = metadata.fps / sample_interval`
   - Passes `effective_fps` to `_get_tracker()` for accurate motion prediction

3. **Added `tracker.reset()` at video start**:
   - Ensures clean tracker state for each new video processed

4. **Added person-only filtering**:
   - New `_filter_persons_only()` helper method
   - Ball detections (class_id=32) are separated before tracking
   - Only person detections go through ByteTrack
   - Balls recombined after court filtering

5. **Both batch processing loops fixed** (main batch and remaining frames)

---

### 3. Court Detector Fixes (`backend/app/ml/court_detector.py`)

**Changes Made:**

1. **Added `_last_mask` instance variable** for caching

2. **Added `_conservative_fallback_mask()` method**:
   - Creates mask covering inner 70% of frame
   - Uses 15% margins on all sides to exclude sidelines/audience

3. **Added `_get_fallback_mask()` helper**:
   - Returns cached mask if available and dimensions match
   - Otherwise returns conservative fallback

4. **Updated `detect_court_mask()`**:
   - On success: caches mask with `self._last_mask = mask.copy()`
   - On failure: uses fallback instead of full-frame mask

---

### 4. Parameter Tuning

**`backend/app/config.py`:**
| Parameter | Old | New |
|-----------|-----|-----|
| `yolo_confidence_threshold` | 0.5 | 0.35 |

**`backend/app/services/detection_pipeline.py` (DetectionPipelineConfig):**
| Parameter | Old | New |
|-----------|-----|-----|
| `tracking_buffer_seconds` | 1.0 | 2.0 |
| `tracking_iou_threshold` | 0.8 | 0.6 |
| `court_overlap_threshold` | 0.3 | 0.2 |

---

### 5. Test Updates (`backend/tests/`)

- Updated `test_ml_byte_tracker.py` to mock `sv.Detections` with all required attributes
- Updated `test_detection_pipeline.py` with missing mock settings

---

## Verification

- **All 274 backend tests pass**
- **No regressions introduced**
- Ready for manual testing with real video to confirm tracking stability

