# bbva-2uy: Batch-Based Pipeline - Summary

**Full context**: See `bbva-2uy.md` for complete details.

## Current Status

**IN PROGRESS** - Core infrastructure complete, integration pending.

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

## Completed

| Component | File | Status |
|-----------|------|--------|
| ProcessingBatch model | `backend/app/models/processing_batch.py` | Done |
| Migration | `backend/alembic/versions/96312ae6fc68_*.py` | Done |
| DetectionBatchProcessor | `backend/app/services/batch_processor.py` | Done |
| OCRBatchProcessor | `backend/app/services/batch_processor.py` | Done |
| SequentialOrchestrator | `backend/app/services/batch_orchestrator.py` | Done |
| Resume logic | In orchestrator | Done |
| Config settings | `backend/app/config.py` | Done |

## Remaining

1. Refactor `detection_pipeline.py` to use orchestrator (or create parallel path)
2. Update job worker to use new orchestrator
3. Write unit tests for batch processors
4. Test resume/interrupt scenarios

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

STATUS: Core infrastructure COMPLETE. Integration pending.

COMPLETED:
- ProcessingBatch model + migration (applied)
- DetectionBatchProcessor and OCRBatchProcessor in batch_processor.py
- SequentialOrchestrator in batch_orchestrator.py with resume logic
- Config settings added

REMAINING:
1. Wire orchestrator into job system (update detection_pipeline.py or job worker)
2. Write unit tests for new batch processors
3. Test resume/interrupt scenarios

KEY FILES:
- backend/app/services/batch_processor.py
- backend/app/services/batch_orchestrator.py
- backend/app/models/processing_batch.py
- backend/app/services/detection_pipeline.py (needs update)

All 34 detection tests still pass.
```
