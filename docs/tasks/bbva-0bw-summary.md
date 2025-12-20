# bbva-0bw: Jersey Number OCR - Summary

**Full context**: See `bbva-0bw.md` for complete details.

## Current Status

**CLOSED** - All sub-issues complete. OCR implementation functional but slow.

## Known Issue: OCR Performance

SmolVLM2 is slow (~15s/frame on Apple Silicon MPS). This blocks detection progress.

**Workaround**: OCR disabled in `backend/.env` (`enable_jersey_ocr=false`)

**Fix**: See [bbva-2uy](./bbva-2uy.md) - Batch-based pipeline with checkpointing
- Separates detection and OCR into independent processors
- Processes in batches with DB checkpoints
- Resumable if interrupted

## Completed Issues

| Issue | Description | Files | Tests |
|-------|-------------|-------|-------|
| bbva-dhn | Legibility filter | `backend/app/ml/legibility_filter.py` | 18 |
| bbva-4f5 | SmolVLM2 OCR | `backend/app/ml/jersey_ocr.py` | 17 |
| bbva-azp | Database model | `backend/app/models/jersey_number.py` + migration | - |
| bbva-0co | Pipeline integration | `backend/app/services/detection_pipeline.py` | 20 |
| bbva-axr | Aggregation service | `backend/app/services/jersey_aggregator.py` | 7 |
| bbva-sbi | API endpoints | `backend/app/api/videos.py` | 29 |

## API Endpoints (New)

```
GET /api/videos/{id}/jersey-numbers
  - Query params: tracking_id (optional), valid_only (bool)
  - Returns: All OCR readings for a video

GET /api/videos/{id}/jersey-numbers/by-track
  - Returns: Aggregated jersey numbers per tracking_id
  - Includes: confidence, conflict detection, all observed numbers
```

## How to Test from UI

1. Process a video with detection enabled (existing flow)
2. OCR runs automatically during detection (samples every 10th frame per track)
3. Fetch results: `GET /api/videos/{video_id}/jersey-numbers/by-track`
4. Display aggregated jersey numbers on the detection overlay

## Dependencies

```toml
transformers = ">=4.49.0"
accelerate = ">=0.27.0"
pillow = ">=10.0.0"
num2words = ">=0.5.13"
```

## Session Restart Prompt

```
Jersey Number OCR implementation is COMPLETE in basketball-video-analyzer.

ALL PHASES DONE:
- Legibility filter: backend/app/ml/legibility_filter.py
- SmolVLM2 OCR: backend/app/ml/jersey_ocr.py
- Database model: backend/app/models/jersey_number.py
- Pipeline integration: backend/app/services/detection_pipeline.py
- Aggregation: backend/app/services/jersey_aggregator.py
- API endpoints: backend/app/api/videos.py (new endpoints)

READY FOR UI TESTING:
1. Process a video via existing detection flow
2. Fetch GET /api/videos/{id}/jersey-numbers/by-track
3. Display jersey numbers on detection overlay in frontend

The parent issue bbva-0bw can be closed once UI integration is verified.
```
