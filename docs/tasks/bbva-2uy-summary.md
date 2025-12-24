# bbva-2uy: Batch-Based Pipeline - Summary

**Full context**: See `bbva-2uy.md` for complete details.

## Current Status

**TRACKING QUALITY IMPROVEMENTS COMPLETE** - Identity switch detection and shoe color extraction implemented.

## Latest Session Accomplishments

### 6. Identity Switch Detection (Complete)
- Created `IdentitySwitchDetector` service in `identity_switch_detector.py`
- Detects when a track's jersey number changes mid-video
- Splits tracks at the transition point by assigning new tracking IDs
- Uses sliding window analysis of OCR readings
- Configurable via env vars: `IDENTITY_SWITCH_WINDOW_SIZE_FRAMES`, `IDENTITY_SWITCH_MIN_READINGS`, `IDENTITY_SWITCH_THRESHOLD`
- Runs BEFORE track merging in the orchestrator pipeline

### 7. Shoe Color Extraction (Complete)
- Added `extract_shoe_color()` function to `color_extractor.py`
- Extracts HSV histogram from bottom 20% of player bounding box
- Added `extract_combined_colors()` for convenience
- Added `shoe_color_hist` field to `Detection` type
- `batch_processor.py` now extracts both jersey and shoe colors

### 8. New Tests Added
- `test_identity_switch_detector.py` - 6 tests for identity switch detection
- `test_color_extractor.py` - 15 tests for jersey/shoe color extraction
- `test_api_detection.py` - 3 new tests for reprocess-tracks endpoint
- All 346 tests pass

### 9. Reprocess API Endpoint (Complete)
- Added `POST /api/videos/{video_id}/reprocess-tracks` endpoint
- Allows running identity switch detection and track merging on already-analyzed videos
- Configurable via request body: `enable_identity_switch_detection`, `enable_track_merging`
- Returns detailed stats: switches detected, tracks before/after merge, spatial/jersey merges

## Previous Session Accomplishments

### 1. Batch Pipeline Integration (Complete)
- Job worker now uses `SequentialOrchestrator`
- Resume/extend logic working

### 2. Parallel OCR Processing (Complete)
- Added `ThreadPoolExecutor` for concurrent OCR
- Configurable via `OCR_MAX_WORKERS` (default: 4)
- Added thread-safe model loading with lock

### 3. Logging & Observability (Complete)
- Added `LOG_LEVEL` env var (default: INFO)
- Added inference timing for detection batches
- Added inference timing for OCR batches (shows workers used)

### 4. Jersey-Based Track Merging (Complete)
- Track merger now queries jersey numbers from OCR data
- Merges non-overlapping tracks with same jersey number
- Logs show: `Merged X → Y tracks (spatial=N, jersey=M)`

### 5. Documentation Updates (Complete)
- Created `docs/architecture.md` with Mermaid diagrams
- Created `docs/diagrams/*.mmd` and `*.svg` files
- Updated `implementation-plan.md` with current tech stack
- Updated `README.md` and `beads-structure.md`

## Proposed Next Steps (Not Yet Implemented)

### Priority 1: Better OCR Model
Consider switching from SmolVLM2 to PaddleOCR for faster, more accurate number recognition.

### Priority 2: Use Shoe Colors for Track Merging
Use shoe color similarity as an additional signal in track merging logic.

## Configuration Reference

```bash
# Batch processing
BATCH_FRAMES_PER_BATCH=30
BATCH_SAMPLE_INTERVAL=3
OCR_MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
ENABLE_INFERENCE_TIMING=true

# Track merging
ENABLE_JERSEY_MERGE=true
MIN_JERSEY_CONFIDENCE=0.6
MIN_JERSEY_READINGS=2

# Identity switch detection
ENABLE_IDENTITY_SWITCH_DETECTION=true
IDENTITY_SWITCH_WINDOW_SIZE_FRAMES=150
IDENTITY_SWITCH_MIN_READINGS=3
IDENTITY_SWITCH_THRESHOLD=0.7
```

## Key Files Modified This Session

| File | Changes |
|------|---------|
| `backend/app/services/identity_switch_detector.py` | NEW - Identity switch detection and track splitting |
| `backend/app/services/batch_orchestrator.py` | Integrated identity switch detection before track merging |
| `backend/app/services/batch_processor.py` | Now extracts both jersey and shoe colors |
| `backend/app/ml/color_extractor.py` | Added shoe color extraction functions |
| `backend/app/ml/types.py` | Added shoe_color_hist field to Detection |
| `backend/app/config.py` | Added identity switch detection settings |
| `backend/app/api/detection.py` | NEW endpoint: POST /videos/{id}/reprocess-tracks |
| `backend/app/schemas/detection.py` | Added TrackReprocessRequest/Response |
| `backend/tests/test_identity_switch_detector.py` | NEW - 6 tests |
| `backend/tests/test_color_extractor.py` | NEW - 15 tests |
| `backend/tests/test_api_detection.py` | Added 3 tests for reprocess-tracks |

## Session Restart Prompt

```
Continue work on basketball-video-analyzer tracking improvements.

CONTEXT:
- Batch pipeline is COMPLETE and working
- Parallel OCR implemented (ThreadPoolExecutor)
- Jersey-based track merging implemented
- Identity switch detection IMPLEMENTED and tested
- Shoe color extraction IMPLEMENTED and tested
- Reprocess API endpoint added for already-analyzed videos
- All 346 tests pass

COMPLETED SOLUTIONS:
1. Identity switch detection - splits tracks when jersey number changes mid-track
2. Shoe color extraction - extracts from bottom 20% of bbox
3. API endpoint: POST /api/videos/{id}/reprocess-tracks

PROPOSED NEXT STEPS:
1. Switch to PaddleOCR for better number recognition
2. Use shoe colors as additional signal in track merging

KEY FILES:
- backend/app/services/identity_switch_detector.py - identity switch detection
- backend/app/services/track_merger.py - jersey merge logic
- backend/app/services/batch_processor.py - OCR and color processing
- backend/app/ml/color_extractor.py - jersey and shoe color extraction
- backend/app/api/detection.py - reprocess-tracks endpoint

API USAGE:
# Reprocess tracks on already-analyzed video
curl -X POST http://localhost:8000/api/videos/{video_id}/reprocess-tracks

# With options
curl -X POST http://localhost:8000/api/videos/{video_id}/reprocess-tracks \
  -H "Content-Type: application/json" \
  -d '{"enable_identity_switch_detection": true, "enable_track_merging": true}'

DIAGNOSTIC QUERIES:
# Check jersey readings per track
SELECT tracking_id, parsed_number, COUNT(*) as readings
FROM jersey_numbers WHERE video_id = 7 AND is_valid = 1
GROUP BY tracking_id, parsed_number ORDER BY tracking_id;

# Check track time ranges
SELECT tracking_id, MIN(frame_number), MAX(frame_number)
FROM player_detections WHERE video_id = 7
GROUP BY tracking_id;
```
