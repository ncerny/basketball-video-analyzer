# SAM3-Only Cleanup Design

**Date**: 2026-01-25
**Status**: Approved
**Goal**: Remove all non-SAM3 detection/tracking code and simplify the codebase

## Context

The basketball video analyzer originally supported multiple detection backends (YOLO, RF-DETR) and tracking backends (ByteTrack, Norfair, SAM2, SAM3). With SAM3 working well for unified detection+tracking via text prompts, the legacy code is no longer needed.

Additionally, the API previously supported both in-process job execution and external workers. Going forward, runners will always be separate processes, so in-process execution code can be removed from the API.

## Scope

### Files to Delete

**ML Models (6 files, ~1,500 lines):**
- `backend/app/ml/yolo_detector.py` - YOLO detection
- `backend/app/ml/rfdetr_detector.py` - RF-DETR detection
- `backend/app/ml/byte_tracker.py` - ByteTrack tracking
- `backend/app/ml/norfair_tracker.py` - Norfair tracking
- `backend/app/ml/sam2_tracker.py` - SAM2 tracking (756 lines)
- `backend/app/ml/sam2_mask_extractor.py` - SAM2 mask extraction (196 lines)

**Services (2 files):**
- `backend/app/services/batch_processor.py` - DetectionBatchProcessor, OCRBatchProcessor
- `backend/app/services/batch_orchestrator.py` - SequentialOrchestrator

**Tests (3 files):**
- `backend/tests/test_ml_yolo_detector.py`
- `backend/tests/test_ml_byte_tracker.py`
- `backend/tests/test_detection_pipeline.py`

### Files to Simplify

**`backend/app/services/detection_pipeline.py`:**
- Remove `DetectionPipeline` class entirely
- Remove `create_detection_job_worker()` function
- Keep only imports/utilities needed by other modules (or delete entirely if nothing needed)

**`backend/app/api/detection.py`:**
- Remove `_ensure_detection_worker_registered()` function
- `start_detection()` - Remove in-process execution path (keep external worker only)
- `get_job_status()` - Remove in-memory job manager fallback
- `cancel_job()` - Remove in-memory job manager fallback
- `get_ml_config()` - Update to return SAM3 config only

**`backend/app/main.py`:**
- Remove worker registration from `startup_event()`

**`backend/app/config.py`:**
- Remove `detection_backend` setting
- Remove `tracking_backend` setting (or change to `Literal["sam3"]`)
- Remove all YOLO settings (lines 29-36)
- Remove all SAM2 settings (lines 63-78)
- Keep SAM3 settings

### Tests to Update

**`backend/tests/test_api_detection.py`:**
- Remove in-process worker registration tests
- Keep external worker tests

**`backend/tests/test_sam3_config.py`:**
- Update to validate SAM3-only configuration

### Dependencies to Remove

From `pyproject.toml`:
- `sam2 = "^1.0.0"` (ml group)
- `ultralytics = "^8.3.0"` (ml group) - YOLO
- `supervision = "^0.23.0"` (ml group) - ByteTrack
- `rfdetr = "^1.3.0"` (api group)
- `norfair = "^2.2.0"` (api group)

### What Remains

**ML directory after cleanup:**
- `sam3_tracker.py` - Core SAM3 video tracker
- `sam3_detection_pipeline.py` - SAM3 pipeline orchestration
- `sam3_frame_extractor.py` - Frame extraction for SAM3
- `cv_utils_fallback.py` - Kernel fallback utilities
- `types.py` - Detection/BoundingBox types

**Config after cleanup:**
- SAM3 settings only (prompt, confidence, precision, memory window, torch.compile)
- No backend selection logic

## Out of Scope

The following are explicitly not part of this cleanup:

1. **Player identity mapping** - Tying tracking IDs to player names/numbers
2. **Game analytics** - Assists, scores, fouls, shot percentage, time on court
3. **OCR replacement** - Alternative approaches to jersey number reading

These are significant features that deserve their own design sessions.

## Implementation Order

1. Delete ML model files (yolo, rfdetr, bytetrack, norfair, sam2)
2. Delete batch_processor.py and batch_orchestrator.py
3. Simplify detection_pipeline.py (remove or gut)
4. Simplify detection.py API endpoints
5. Clean up main.py startup
6. Simplify config.py
7. Remove dependencies from pyproject.toml
8. Delete legacy test files
9. Update remaining tests
10. Run full test suite
11. Test cloud worker deployment

## Risks

- **Breaking changes**: This removes configuration options that may be referenced elsewhere
- **Mitigation**: Search for all usages of removed config keys before deletion

- **Worker compatibility**: Ensure cloud worker still works after cleanup
- **Mitigation**: Test deployment after changes

## Success Criteria

- [ ] All legacy ML files deleted
- [ ] detection_pipeline.py simplified or removed
- [ ] API endpoints only support external workers
- [ ] Config has only SAM3 settings
- [ ] Dependencies removed from pyproject.toml
- [ ] All tests pass
- [ ] Cloud worker deploys and processes videos successfully
