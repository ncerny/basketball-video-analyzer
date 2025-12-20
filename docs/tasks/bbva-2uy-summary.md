# bbva-2uy: Batch-Based Pipeline - Summary

**Full context**: See `bbva-2uy.md` for complete details.

## Current Status

**NOT STARTED** - Design complete, ready for implementation.

## Problem

- Detection processes entire video before writing to DB
- If interrupted at frame 500/900, all progress lost
- OCR runs inline (~15s/frame), blocking detection

## Solution

Process in batches with DB checkpoints:

```
Batch 1: Detect frames 0-29 → write to DB → OCR batch 1 → write to DB
Batch 2: Detect frames 30-59 → write to DB → OCR batch 2 → write to DB
...
```

## Key Components to Build

1. `ProcessingBatch` model + migration
2. `DetectionBatchProcessor` class
3. `OCRBatchProcessor` class
4. `SequentialOrchestrator` class
5. Resume logic (query incomplete batches, skip completed)

## Files to Create/Modify

| File | Action |
|------|--------|
| `backend/app/models/processing_batch.py` | CREATE |
| `backend/alembic/versions/*_add_processing_batches.py` | CREATE |
| `backend/app/services/batch_processor.py` | CREATE |
| `backend/app/services/batch_orchestrator.py` | CREATE |
| `backend/app/services/detection_pipeline.py` | MODIFY |
| `backend/app/config.py` | MODIFY (add batch settings) |

## Session Restart Prompt

```
Continue work on basketball-video-analyzer batch pipeline (bbva-2uy).

GOAL: Refactor detection/OCR into batch-based processors with checkpointing.

DESIGN: See docs/tasks/bbva-2uy.md and docs/implementation-plan.md section 2.4

CURRENT STACK (unchanged):
- Detection: RF-DETR
- Tracking: Norfair  
- OCR: SmolVLM2

NEXT STEPS:
1. Create ProcessingBatch model
2. Create migration
3. Implement DetectionBatchProcessor
4. Implement OCRBatchProcessor
5. Implement SequentialOrchestrator
6. Refactor detection_pipeline.py to use orchestrator
```
