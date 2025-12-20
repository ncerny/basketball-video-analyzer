# bbva-2uy: Batch-Based Pipeline with Checkpointing

## Problem Statement

Current detection pipeline processes entire videos in one go:
- All frames processed in memory
- Single DB commit at the end
- If interrupted at frame 500 of 900, all progress lost
- OCR runs inline during detection, blocking progress (~15s/frame)

## Goals

1. **Resilience**: Process in batches, checkpoint to DB after each batch
2. **Resumability**: If interrupted, resume from last completed batch
3. **Separation**: Detection and OCR as independent processors (DAG-style dependencies)
4. **Flexibility**: Architecture supports local sequential execution now, distributed later

## Architecture Design

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

### Execution Modes

```python
# Mode 1: Local Sequential (current priority)
# Minimizes resource contention, predictable performance
for batch in batches:
    detect(batch)      # uses GPU/MPS
    commit()           # checkpoint
    ocr(batch)         # uses GPU/MPS  
    commit()           # checkpoint

# Mode 2: Local Pipeline (future, if CPU cycles available)
# Detection batch N+1 runs while OCR processes batch N
detection_queue = Queue()
Thread 1: detect(batch) → push to queue → detect(next batch)
Thread 2: pop from queue → ocr(batch)

# Mode 3: Distributed (future, separate nodes)
# Detection node creates OCR tasks, OCR node picks them up
Detection Node: detect(batch) → commit → enqueue_ocr_task(batch_id)
OCR Node:       poll_tasks() → ocr(batch) → commit
```

### Database Schema Addition

```sql
-- New table: processing_batches
CREATE TABLE processing_batches (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    batch_index INTEGER,              -- 0, 1, 2, ...
    frame_start INTEGER,
    frame_end INTEGER,
    detection_status TEXT,            -- pending, processing, completed, failed
    ocr_status TEXT,                  -- pending, processing, completed, failed, skipped
    detection_completed_at TIMESTAMP,
    ocr_completed_at TIMESTAMP,
    created_at TIMESTAMP
);

-- Index for resume queries
CREATE INDEX idx_processing_batches_video_status 
ON processing_batches(video_id, detection_status);
```

### Code Structure

```
backend/app/services/
├── detection_pipeline.py      # Current monolith (to be refactored)
├── batch_processor.py         # NEW: Individual batch operations
│   ├── DetectionBatchProcessor
│   └── OCRBatchProcessor
├── batch_orchestrator.py      # NEW: Coordinates batch execution
│   ├── SequentialOrchestrator    # Local sequential (Mode 1)
│   ├── PipelineOrchestrator      # Local pipeline (Mode 2)  
│   └── DistributedOrchestrator   # Future (Mode 3)
└── job_manager.py             # Existing job system
```

### Config Additions

```python
# config.py additions
batch_size_frames: int = 30           # Frames per detection batch
execution_mode: Literal["sequential", "pipeline", "distributed"] = "sequential"
ocr_enabled: bool = True
```

## Implementation Plan

### Phase 1: Batch Infrastructure
1. Create `ProcessingBatch` SQLAlchemy model + migration
2. Create `DetectionBatchProcessor` class (extracts batch logic from current pipeline)
3. Create `OCRBatchProcessor` class (extracts OCR logic from current pipeline)
4. Create `SequentialOrchestrator` that coordinates batches

### Phase 2: Resume Logic
1. Add `get_incomplete_batches(video_id)` query
2. On job start, check for existing batches and resume
3. Handle edge cases (partial batches, failed batches)

### Phase 3: Refactor Detection Pipeline
1. Update `DetectionPipeline.process_video()` to use orchestrator
2. Update progress reporting to reflect batch progress
3. Maintain backward compatibility with existing job API

### Phase 4: Testing
1. Unit tests for batch processors
2. Integration tests for resume scenarios
3. Test interruption recovery

## Progress Tracking

### Improved Progress UX

Current issue: Percentage shows pipeline progress, message shows frame progress (confusing).

New approach:
- Report frame-based progress within detection phase
- Report detection-based progress within OCR phase

```python
# Detection phase: frames processed
f"Detection: {frames_done}/{total_frames} frames ({pct}%)"

# OCR phase: detections processed  
f"OCR: {detections_done}/{total_detections} ({pct}%)"

# Overall: combine phases
overall = (detection_weight * detection_pct + ocr_weight * ocr_pct) / 100
```

## Hardware Considerations

### Current Performance (SmolVLM2)
| Device | Per-frame OCR | Notes |
|--------|---------------|-------|
| MPS (M1/M2) | ~15s | Current, too slow |
| NVIDIA CUDA | ~2-4s | Estimated |
| AMD ROCm | ~3-5s | Experimental support |
| CPU | ~30s+ | Not viable |

### Future Optimizations (not in scope)
- Batch OCR inference (process multiple crops at once)
- Smaller/faster OCR models
- MLX-optimized models for Apple Silicon

## Success Criteria

1. Detection can be interrupted and resumed without losing progress
2. OCR can be interrupted and resumed without losing progress
3. Progress percentage accurately reflects work done
4. No performance regression in happy path (uninterrupted processing)
5. Architecture allows future migration to distributed execution

## Session Restart Prompt

```
Continue work on basketball-video-analyzer batch pipeline (bbva-2uy).

CONTEXT:
- Refactoring detection/OCR into batch-based processors with checkpointing
- See docs/tasks/bbva-2uy.md for full design
- See docs/implementation-plan.md section "Batch-Based Processing Pipeline"

CURRENT STACK (unchanged):
- Detection: RF-DETR
- Tracking: Norfair  
- OCR: SmolVLM2

KEY FILES:
- backend/app/services/detection_pipeline.py (refactor target)
- backend/app/models/ (add ProcessingBatch)
- backend/app/services/batch_processor.py (new)
- backend/app/services/batch_orchestrator.py (new)

GOAL: Resilient, resumable processing with checkpoint after each batch.
```
