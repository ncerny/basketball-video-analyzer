# bbva-2uy: Batch-Based Pipeline with Checkpointing

## Status: COMPLETE

Batch pipeline is fully implemented. Current focus has shifted to tracking quality improvements.

## Problem Statement

Current detection pipeline processes entire videos in one go:
- All frames processed in memory
- Single DB commit at the end
- If interrupted at frame 500 of 900, all progress lost
- OCR runs inline during detection, blocking progress (~15s/frame)

## Solution Implemented

### Batch-Based Task DAG

```
Video Processing Job
│
├─ Batch 1: [Detect frames 0-29] ──→ [OCR batch 1 detections]
├─ Batch 2: [Detect frames 30-59] ──→ [OCR batch 2 detections]  
├─ Batch 3: [Detect frames 60-89] ──→ [OCR batch 3 detections]
└─ ...

Each arrow = dependency (OCR blocked only by its detection batch)
Batches are independent of each other
```

### Key Features

1. **Checkpointing**: DB commit after each batch
2. **Resume**: Find incomplete batches, continue from there
3. **Extend**: Adding time range extends existing batches (doesn't restart)
4. **Parallel OCR**: ThreadPoolExecutor processes multiple crops concurrently
5. **Jersey-based merging**: Post-processing merges tracks with same jersey number

### Database Schema

```sql
CREATE TABLE processing_batches (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    batch_index INTEGER,
    frame_start INTEGER,
    frame_end INTEGER,
    detection_status TEXT,  -- pending, processing, completed, failed
    ocr_status TEXT,        -- pending, processing, completed, failed, skipped
    detection_completed_at TIMESTAMP,
    ocr_completed_at TIMESTAMP,
    created_at TIMESTAMP
);
```

### Code Structure

```
backend/app/services/
├── batch_processor.py         # Individual batch operations
│   ├── DetectionBatchProcessor   # Detection + tracking for frame range
│   └── OCRBatchProcessor         # Parallel OCR for batch detections
├── batch_orchestrator.py      # Coordinates batch execution
│   └── SequentialOrchestrator    # Local sequential execution
├── track_merger.py            # Post-processing track consolidation
│   ├── Spatial merging           # Merge by proximity/timing
│   └── Jersey merging            # Merge by same jersey number
└── detection_pipeline.py      # Job worker uses orchestrator
```

### Configuration

```python
# config.py
batch_frames_per_batch: int = 30
batch_sample_interval: int = 3
batch_execution_mode: Literal["sequential", "pipeline", "distributed"] = "sequential"
ocr_max_workers: int = 4
log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
enable_inference_timing: bool = False

# Track merger
enable_jersey_merge: bool = True
min_jersey_confidence: float = 0.6
min_jersey_readings: int = 2
```

## Implementation Complete

| Component | File | Status |
|-----------|------|--------|
| ProcessingBatch model | `models/processing_batch.py` | Done |
| Migration | `alembic/versions/96312ae6fc68_*.py` | Applied |
| DetectionBatchProcessor | `services/batch_processor.py` | Done |
| OCRBatchProcessor | `services/batch_processor.py` | Done |
| Parallel OCR | `services/batch_processor.py` | Done |
| SequentialOrchestrator | `services/batch_orchestrator.py` | Done |
| Resume logic | `services/batch_orchestrator.py` | Done |
| Extend logic | `services/batch_orchestrator.py` | Done |
| Jersey-based merging | `services/track_merger.py` | Done |
| Job worker integration | `services/detection_pipeline.py` | Done |
| Logging config | `main.py`, `config.py` | Done |
| Inference timing | `batch_processor.py` | Done |

---

## Current Focus: Tracking Quality

### Problem Identified

Analysis revealed the tracking issue is **identity switches**, not fragmentation:
- Tracks run full video duration (no fragmentation)
- But a single track follows different players at different times
- Example: Track 3 has jersey #2 readings (frames 1-400) and #24 readings (frames 500-900)

### Evidence

```
Track ID | Detections | First Frame | Last Frame | Duration
       3 |        809 |           1 |        898 | 897  (full video)

Track 3 jersey readings:
  #24: 9 readings (frames 31-526)
  #12: 8 readings (frames 61-894)
  #22: 1 reading
  #20: 1 reading
```

The track follows player #24, then switches to player #12 mid-video.

### Proposed Solutions

#### 1. Detect & Split Identity Switches (Recommended First)
Use OCR data to detect jersey number changes:
```python
# If track has high-confidence readings for different numbers
# at different time periods, split the track at transition point
Track 3: #24 (frames 31-400) → SPLIT → #12 (frames 500-894)
         becomes Track 3a and Track 3b
```

#### 2. Add Shoe Color Extraction
Expand `color_extractor.py` to include shoes:
- Extract color from bottom 20% of bbox
- Youth players often have unique shoe colors
- Use for identity verification during tracking

#### 3. Switch to PaddleOCR
Replace SmolVLM2 with PaddleOCR:
- Faster inference (~100ms vs ~15s per crop)
- Better accuracy for printed numbers
- Easier to fine-tune

#### 4. Train Custom OCR (Long-term)
- Collect labeled jersey number crops from failures
- Fine-tune model on basketball-specific data

## Session Restart Prompt

```
Continue work on basketball-video-analyzer tracking improvements (bbva-2uy).

STATUS: Batch pipeline COMPLETE. Focusing on identity switch detection.

PROBLEM: Tracks follow different players at different times
- Track 3: jersey #24 → #12 mid-video (identity switch)
- 19 tracks, 6 with jersey numbers, 0 merges (correct - no fragmentation)
- OCR reads multiple jerseys per track (confirms switches)

NEXT STEPS (in priority order):
1. Implement identity switch detection using OCR data
   - Analyze jersey readings over time per track
   - If significant jersey change, split track at transition
2. Add shoe color extraction (bottom 20% of bbox)
3. Consider PaddleOCR for faster/better number recognition

KEY FILES:
- backend/app/services/track_merger.py - add split logic here
- backend/app/services/batch_processor.py - OCR processing  
- backend/app/ml/color_extractor.py - add shoe colors
- backend/app/ml/jersey_ocr.py - current OCR model

DIAGNOSTIC SQL:
-- Jersey readings per track (identity switches visible here)
SELECT tracking_id, parsed_number, COUNT(*), 
       MIN(frame_number), MAX(frame_number)
FROM jersey_numbers 
WHERE video_id = 7 AND is_valid = 1 AND parsed_number IS NOT NULL
GROUP BY tracking_id, parsed_number
ORDER BY tracking_id, COUNT(*) DESC;

All 322 tests pass.
```
