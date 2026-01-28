# SAM3-Only Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all non-SAM3 detection/tracking code and simplify the codebase to SAM3-only.

**Architecture:** Delete legacy ML models (YOLO, RF-DETR, ByteTrack, Norfair, SAM2), remove in-process job execution from API, simplify config to SAM3-only settings.

**Tech Stack:** Python, FastAPI, PyTorch, Transformers (SAM3)

---

### Task 1: Delete Legacy ML Model Files

**Files:**
- Delete: `backend/app/ml/yolo_detector.py`
- Delete: `backend/app/ml/rfdetr_detector.py`
- Delete: `backend/app/ml/byte_tracker.py`
- Delete: `backend/app/ml/norfair_tracker.py`
- Delete: `backend/app/ml/sam2_tracker.py`
- Delete: `backend/app/ml/sam2_mask_extractor.py`

**Step 1: Delete the files**

```bash
rm backend/app/ml/yolo_detector.py
rm backend/app/ml/rfdetr_detector.py
rm backend/app/ml/byte_tracker.py
rm backend/app/ml/norfair_tracker.py
rm backend/app/ml/sam2_tracker.py
rm backend/app/ml/sam2_mask_extractor.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy ML model files (YOLO, RF-DETR, ByteTrack, Norfair, SAM2)"
```

---

### Task 2: Delete Legacy Service Files

**Files:**
- Delete: `backend/app/services/batch_processor.py`
- Delete: `backend/app/services/batch_orchestrator.py`
- Delete: `backend/app/services/detection_pipeline.py`

**Step 1: Delete the files**

```bash
rm backend/app/services/batch_processor.py
rm backend/app/services/batch_orchestrator.py
rm backend/app/services/detection_pipeline.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy service files (batch_processor, batch_orchestrator, detection_pipeline)"
```

---

### Task 3: Delete Legacy Test Files

**Files:**
- Delete: `backend/tests/test_ml_yolo_detector.py`
- Delete: `backend/tests/test_ml_byte_tracker.py`
- Delete: `backend/tests/test_detection_pipeline.py`

**Step 1: Delete the files**

```bash
rm backend/tests/test_ml_yolo_detector.py
rm backend/tests/test_ml_byte_tracker.py
rm backend/tests/test_detection_pipeline.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "test: delete legacy test files for removed ML models and pipeline"
```

---

### Task 4: Simplify Config - Remove Legacy Settings

**Files:**
- Modify: `backend/app/config.py`

**Step 1: Read current config**

Review the file to understand current structure.

**Step 2: Remove legacy settings**

Remove:
- `detection_backend` setting
- `tracking_backend` setting (or simplify to SAM3-only)
- All YOLO settings (yolo_model_name, yolo_confidence_threshold, yolo_person_class_id, yolo_batch_size_*)
- All SAM2 settings (sam2_model_name, sam2_new_object_iou_threshold, sam2_lost_track_frames, sam2_max_memory_frames, sam2_auto_download, sam2_embedding_similarity_threshold, sam2_color_tiebreaker_threshold, sam2_reidentification_enabled)

Keep:
- All SAM3 settings
- All non-ML settings (database, storage, etc.)

**Step 3: Commit**

```bash
git add backend/app/config.py
git commit -m "config: remove legacy detection/tracking backend settings"
```

---

### Task 5: Simplify API Detection Endpoints

**Files:**
- Modify: `backend/app/api/detection.py`

**Step 1: Read current file**

Review to understand the dual-mode (in-process vs external worker) logic.

**Step 2: Remove in-process execution code**

Remove:
- `_ensure_detection_worker_registered()` function
- In-process path from `start_detection()` endpoint
- In-memory job manager fallback from `get_job_status()` endpoint
- In-memory job manager fallback from `cancel_job()` endpoint
- Update `get_ml_config()` to return SAM3 config only

Keep:
- External worker (DB-backed) paths for all endpoints

**Step 3: Remove unused imports**

Remove imports that are no longer needed after cleanup.

**Step 4: Commit**

```bash
git add backend/app/api/detection.py
git commit -m "api: simplify detection endpoints to external worker only"
```

---

### Task 6: Clean Up main.py Startup

**Files:**
- Modify: `backend/app/main.py`

**Step 1: Read current file**

Review the startup_event() function.

**Step 2: Remove worker registration**

Remove:
- Import of `create_detection_job_worker`
- Import of `get_job_manager` (if only used for worker registration)
- Worker registration code in `startup_event()`

**Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "startup: remove in-process worker registration"
```

---

### Task 7: Remove Dependencies from pyproject.toml

**Files:**
- Modify: `backend/pyproject.toml`

**Step 1: Read current file**

Review dependency groups.

**Step 2: Remove legacy dependencies**

Remove from ml group:
- `sam2`
- `ultralytics`
- `supervision`

Remove from api group:
- `rfdetr`
- `norfair`

**Step 3: Commit**

```bash
git add backend/pyproject.toml
git commit -m "deps: remove legacy ML dependencies (sam2, ultralytics, supervision, rfdetr, norfair)"
```

---

### Task 8: Update Dockerfile.worker Filter

**Files:**
- Modify: `backend/Dockerfile.worker`

**Step 1: Read current file**

Review the grep filter for requirements.

**Step 2: Update filter**

The filter already excludes sam2. Remove it from the exclusion list since sam2 won't be in requirements anymore (or leave it for safety - won't hurt).

No changes needed if dependencies are properly removed from pyproject.toml.

**Step 3: Commit (if changes made)**

```bash
git add backend/Dockerfile.worker
git commit -m "docker: update requirements filter for SAM3-only deps"
```

---

### Task 9: Fix Import Errors

**Files:**
- Various files that may import deleted modules

**Step 1: Run import check**

```bash
cd backend && python -c "from app.main import app" 2>&1
```

**Step 2: Fix any import errors**

Search for and remove any remaining imports of deleted modules:
- `from app.ml.yolo_detector import ...`
- `from app.ml.rfdetr_detector import ...`
- `from app.ml.byte_tracker import ...`
- `from app.ml.norfair_tracker import ...`
- `from app.ml.sam2_tracker import ...`
- `from app.ml.sam2_mask_extractor import ...`
- `from app.services.batch_processor import ...`
- `from app.services.batch_orchestrator import ...`
- `from app.services.detection_pipeline import ...`

**Step 3: Commit**

```bash
git add -A
git commit -m "fix: remove imports of deleted modules"
```

---

### Task 10: Update test_api_detection.py

**Files:**
- Modify: `backend/tests/test_api_detection.py`

**Step 1: Read current file**

Review tests for in-process worker logic.

**Step 2: Remove in-process worker tests**

Remove tests that:
- Test `_ensure_detection_worker_registered()`
- Test in-memory job manager paths
- Mock legacy detector/tracker selection

Keep tests that:
- Test external worker job submission
- Test job status from database
- Test job cancellation via database

**Step 3: Commit**

```bash
git add backend/tests/test_api_detection.py
git commit -m "test: update detection API tests for external worker only"
```

---

### Task 11: Update test_sam3_config.py

**Files:**
- Modify: `backend/tests/test_sam3_config.py`

**Step 1: Read current file**

Review tests for tracking_backend validation.

**Step 2: Update tests**

Update `test_tracking_backend_includes_sam3` to validate that:
- Either tracking_backend setting is removed entirely
- Or tracking_backend only allows "sam3"

**Step 3: Commit**

```bash
git add backend/tests/test_sam3_config.py
git commit -m "test: update config tests for SAM3-only backend"
```

---

### Task 12: Run Full Test Suite

**Step 1: Run tests**

```bash
cd backend && python -m pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

Address any test failures caused by the cleanup.

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test failures from SAM3-only cleanup"
```

---

### Task 13: Verify Cloud Worker Build

**Step 1: Build Docker image**

```bash
cd backend && DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f Dockerfile.worker -t basketball-analyzer-worker:test .
```

**Step 2: Verify image runs**

```bash
docker run --rm basketball-analyzer-worker:test python -c "from app.ml.sam3_tracker import SAM3VideoTracker; print('SAM3 imports OK')"
```

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: ensure cloud worker builds with SAM3-only deps"
```

---

### Task 14: Final Cleanup and Documentation

**Step 1: Search for any remaining references**

```bash
grep -r "sam2\|bytetrack\|norfair\|yolo\|rfdetr" backend/app --include="*.py" | grep -v __pycache__
```

**Step 2: Update any stale comments or docstrings**

Remove references to legacy backends in comments.

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: clean up stale references to legacy backends"
```

---

## Verification Checklist

After completing all tasks:

- [ ] All legacy ML files deleted (6 files)
- [ ] All legacy service files deleted (3 files)
- [ ] All legacy test files deleted (3 files)
- [ ] Config simplified to SAM3-only
- [ ] API endpoints use external worker only
- [ ] main.py startup cleaned up
- [ ] Dependencies removed from pyproject.toml
- [ ] All tests pass
- [ ] Docker image builds successfully
- [ ] No remaining imports of deleted modules
