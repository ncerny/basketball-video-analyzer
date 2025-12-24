# Basketball Video Analyzer - Development Progress

**Last Updated**: 2025-12-19
**Current Phase**: Phase 2 - Computer Vision Integration
**Project Status**: 47/59 issues closed (80% complete)

---

## 🎯 Project Overview

Basketball Video Analyzer is a local-first application for analyzing youth basketball games. The goal is to enable parents and coaches to:
- Analyze full games in <1 hour
- Find specific player plays in <2 minutes
- Generate highlight reels in <5 minutes

### Tech Stack
- **Frontend**: React 18+, TypeScript, Mantine UI, Zustand, Vite
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, SQLite
- **ML/Video**: YOLOv8-nano, ByteTrack, OpenCV, FFmpeg
- **Issue Tracking**: Beads CLI (`bd`)
- **Git Workflow**: Trunk-based (direct commits to `main`)

---

## 📊 Current Status

### Phase 2: Computer Vision Integration (Epic: bbva-g2m)

**Completed Features** ✅:
1. **Detection Pipeline (bbva-ntd)** - YOLOv8-nano person/ball detection
2. **Player Tracking (bbva-43e)** - ByteTrack for persistent IDs across frames
3. **Detection API (bbva-ck1)** - FastAPI endpoints for detection jobs
4. **Detection Overlay UI (bbva-unh)** - SVG bounding boxes on video player
5. **Detection Trigger UI (bbva-90h)** - Run detection button with progress tracking
6. **Court Detection (bbva-l8b)** - Spatial filtering to remove audience detections
7. **GPU Acceleration (bbva-icp)** - Device-optimized batch sizes and performance logging

**Recent Achievement**: GPU acceleration now working on Apple Silicon (MPS)
- Auto-detects CUDA → MPS → CPU
- Optimized batch sizes: CPU=8, MPS=16, CUDA=32
- Performance logging enabled
- Expected 2-3x speedup on MPS vs CPU

### Remaining Phase 2 Work (Priority 1):

**Core Features**:
1. **bbva-0bw**: Jersey Number OCR Service
   - Extract jersey numbers from player detections
   - Use EasyOCR/Tesseract with preprocessing
   - Store results with confidence scores

2. **bbva-a53**: Player-Detection Matching Service (blocks on: bbva-0bw)
   - Link detections to roster by jersey number
   - Majority voting across frames
   - Handle ambiguous cases

3. **bbva-eur**: Detection Review UI (blocks on: bbva-a53)
   - Manual correction interface
   - Edit player assignments, jersey numbers
   - Review unmatched detections

**Testing**:
4. **bbva-d3c**: ML Pipeline Unit Tests
   - YOLO, ByteTrack, CourtDetector tests
   - Target >80% code coverage
   - Mock models to avoid downloads

5. **bbva-czt**: End-to-End CV Testing (blocks on: bbva-eur)
   - Full pipeline validation
   - Ground truth comparison
   - Performance benchmarks

### Enhancement Tasks (Priority 2):
- **bbva-5gm**: Enhanced processing status UI with stage-specific progress
- **bbva-6ou**: Detection accuracy improvements (fine-tuning, temporal smoothing)

---

## 🏗️ Architecture Highlights

### Multi-Video Timeline (Most Complex Feature)
- Multiple videos from different cameras stitched into continuous game timeline
- Videos have `recorded_at`, `sequence_order`, `game_time_offset`
- Annotations reference unified `game_timestamp`, not video-specific timestamps
- Cross-video segments via `annotation_videos` junction table

### ML Pipeline
```
Video Upload → Frame Extraction → YOLO Detection → ByteTrack Tracking
              ↓                                    ↓
         Court Detection ← Spatial Filtering ← Detections
              ↓
         Database Storage
```

**Key Optimizations**:
- asyncio.to_thread() for CPU-intensive operations (keeps API responsive)
- Device-specific batch sizes for GPU acceleration
- Court detection filters ~50% of false positives (audience members)
- Sample every 3rd frame to reduce processing time

### Database Schema
```
games → videos (1:many)
games ↔ players via game_rosters (many:many)
annotations ↔ videos via annotation_videos (many:many)
annotations → plays (1:1)
videos → player_detections (1:many)
```

---

## 🔧 Recent Technical Work

### Session: GPU Acceleration Implementation (bbva-icp)

**Problem**: Detection running slowly on CPU despite GPU availability

**Solution Implemented**:
1. **Config Changes** (`backend/app/config.py`):
   - Added device-specific batch sizes
   - Added `enable_inference_timing` flag

2. **Pipeline Updates** (`backend/app/services/detection_pipeline.py`):
   - `_get_optimal_batch_size()` method for device-aware sizing
   - Performance logging with time tracking
   - Logs FPS and batch timing when enabled

3. **Model Loading** (`backend/app/ml/yolo_detector.py`):
   - Added logging on model load showing device
   - Confirms GPU usage at startup

4. **Tools Created**:
   - `backend/scripts/benchmark_devices.py` - Compare CPU/MPS/CUDA performance
   - `backend/scripts/check_device.py` - Diagnostic script for device detection
   - `/api/ml-config` endpoint - Runtime device configuration check

**Current Configuration** (verified via API):
```json
{
  "resolved_device": "mps",
  "batch_size": 16,
  "inference_timing_enabled": true,
  "torch": {
    "version": "2.9.1",
    "mps_available": true
  }
}
```

**Files Modified**:
- `backend/app/config.py`
- `backend/app/services/detection_pipeline.py`
- `backend/app/ml/yolo_detector.py`
- `backend/app/api/detection.py`

**Files Created**:
- `backend/scripts/benchmark_devices.py`
- `backend/scripts/check_device.py`

---

## 🚀 Next Steps

### Immediate Priorities:
1. **Jersey OCR** (bbva-0bw) - Foundation for player identification
2. **Unit Tests** (bbva-d3c) - Can start immediately (no blockers)

### Parallel Work Available:
- Enhanced status UI (bbva-5gm)
- Detection accuracy improvements (bbva-6ou)

### Sequential Pipeline:
```
OCR (bbva-0bw) → Player Matching (bbva-a53) → Review UI (bbva-eur) → E2E Tests (bbva-czt)
```

---

## 📝 Development Workflow

```bash
# 1. Check available work
bd ready

# 2. Claim issue
bd update <id> --status=in_progress

# 3. Make changes and test

# 4. Commit (trunk-based)
git commit -m "type(scope): [bbva-xxx] description"
git push origin main

# 5. Close issue
bd close <id> --reason="description"

# 6. Sync beads
bd sync
```

---

## 🐛 Recent Fixes

**Detection Performance Issues** (Multiple PRs):
- Fixed index out of bounds in ByteTrack when detections filtered
- Added asyncio.to_thread() to prevent API blocking during inference
- Fixed startup hanging by making worker registration non-blocking
- Added missing asyncio import
- Fixed frontend timeout errors (increased to 120s)
- Added /api prefix to frontend API client

**Detection Quality** (bbva-l8b):
- Implemented court detection with Canny edge + Hough line transform
- 30% bbox overlap threshold filters audience members
- Runs in thread pool to avoid blocking

---

## 📚 Key Resources

- **Agent Instructions**: `AGENTS.md` (start here)
- **Implementation Plan**: `docs/implementation-plan.md`
- **Task Structure**: `docs/beads-structure.md`
- **Contributing Guide**: `docs/CONTRIBUTING.md`

---

## 💡 Important Notes

### Multi-Video Timeline Pattern
Always reference `game_timestamp` (unified time) in annotations, never video-specific timestamps. This enables cross-video segments and consistent playback across camera angles.

### Beads (Issue Tracker)
- Use `bd` CLI, NOT GitHub Issues
- All tasks tracked in `.beads/` directory
- Sync with `bd sync` after closing issues

### GPU Acceleration
- Auto-detection works: CUDA → MPS → CPU
- Enable logging: `ENABLE_INFERENCE_TIMING=true` in `.env`
- Check config: `curl http://localhost:8000/api/ml-config`
- Run benchmark: `python backend/scripts/benchmark_devices.py`

### Performance Targets
- Full game analysis: <1 hour
- Find player play: <2 minutes
- Generate highlight reel: <5 minutes
- Detection speed: 2-3x faster on MPS vs CPU

---

**For detailed technical specs, see `docs/implementation-plan.md`**
**For current tasks, run `bd ready` or `bd list --status=open`**
