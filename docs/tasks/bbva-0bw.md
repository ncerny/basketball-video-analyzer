# bbva-0bw: Implement Jersey Number OCR Service

**Status**: In Progress  
**Priority**: P1  
**Type**: Feature  
**Created**: 2025-12-19  
**Updated**: 2025-12-20  
**Blocks**: bbva-a53 (Player-detection matching service)

## Problem Statement

Current tracking (RF-DETR + Norfair + Color Re-ID) achieves 15 tracks for ~10 players. However, players wearing the same jersey color (same team) still occasionally swap track IDs during crossings. HSV color histograms cannot distinguish players on the same team.

**Solution**: Read jersey numbers via OCR to definitively identify players. Jersey number is the authoritative player identifier.

## Context From Previous Work (bbva-c6k)

| Stage | Unique Tracks | Technology |
|-------|---------------|------------|
| YOLO + ByteTrack | 133 | IOU matching |
| RF-DETR + ByteTrack | 133 | Better detection |
| RF-DETR + Norfair | 40 | Euclidean distance |
| RF-DETR + Norfair + Color Re-ID | 15 | HSV histogram |

Tracking is now "good enough" - OCR will provide ground truth to:
1. Confirm correct tracking
2. Detect and correct ID swaps
3. Build player-to-track associations

---

## Technology Decision

### Research Summary

| Criteria | SmolVLM2 | ResNet | PaddleOCR | EasyOCR |
|----------|----------|--------|-----------|---------|
| **Type** | Vision-Language Model | CNN Classification | Traditional OCR | Traditional OCR |
| **macOS/MPS** | ✅ MLX native | ⚠️ Partial | ❌ Freezes | ⚠️ CPU only |
| **Pre-trained** | No (fine-tune ~80 imgs) | No (needs thousands) | No | No |
| **Accuracy** | 86% base, higher tuned | 93% (requires training) | Good general | Weak on digits |
| **Speed** | ~8-10s/frame | <10ms | Fast | Moderate |

### Decision: SmolVLM2 + Heuristic Filters

**Why not ResNet?** Requires labeled training data (legible vs illegible crops) we don't have.

**Approach**: Use rule-based legibility filters instead of ResNet to avoid wasting OCR on unreadable crops:

| Filter | Purpose | Implementation |
|--------|---------|----------------|
| Min bbox size | Skip tiny crops | `width * height > threshold` |
| Blur detection | Skip motion blur | Laplacian variance |
| Confidence score | Skip weak detections | `confidence > 0.7` |
| Aspect ratio | Skip weird crops | `height > width` |

**Phase 2 (if needed)**: Collect failure cases from Phase 1, train ResNet legibility classifier.

---

## Detailed Implementation Plan

### Phase 1: Legibility Filter & SmolVLM2 Setup

#### Task 1.1: Create legibility filter module
- File: `backend/app/ml/legibility_filter.py`
- Implement heuristic checks (size, blur, confidence, aspect ratio)
- Return legibility score 0-1
- Unit tests

#### Task 1.2: Create SmolVLM2 OCR service
- File: `backend/app/ml/jersey_ocr.py`
- Install: `transformers`, MLX dependencies
- Prompt engineering for number extraction
- Handle model loading/caching
- Unit tests with sample crops

#### Task 1.3: Create database model
- File: `backend/app/models/jersey_number.py`
- Fields: detection_id, tracking_id, frame_number, raw_ocr_output, parsed_number, confidence, is_valid
- Migration file

### Phase 2: Pipeline Integration

#### Task 2.1: Integrate with detection pipeline
- Modify: `backend/app/services/detection_pipeline.py`
- Run OCR on sampled frames (every Nth frame per track)
- Apply legibility filter before OCR
- Run in thread pool (asyncio.to_thread)
- Store results to database

#### Task 2.2: Create OCR aggregation service
- File: `backend/app/services/jersey_aggregator.py`
- Aggregate OCR results per tracking_id
- Voting/confidence weighting for final number
- Detect conflicting reads (potential swap)

### Phase 3: Player Identification

#### Task 3.1: Track-to-player mapping
- File: `backend/app/services/player_identifier.py`
- Map tracking_id → player by jersey number
- Persist to game_rosters or new junction table
- Handle number collisions (same number, different tracks)

#### Task 3.2: Swap detection
- Detect when track shows different numbers over time
- Log swap events with timestamps
- API to retrieve swap events

### Phase 4: API & Testing

#### Task 4.1: Add API endpoints
- GET `/api/videos/{id}/jersey-numbers` - Get all OCR results
- GET `/api/videos/{id}/tracks/{track_id}/jersey-number` - Get aggregated number for track
- POST `/api/videos/{id}/run-ocr` - Manual OCR trigger

#### Task 4.2: Integration tests
- End-to-end pipeline test
- Accuracy measurement on test video

---

## Sub-Issues

| ID | Title | Phase | Priority |
|----|-------|-------|----------|
| bbva-0bw-1 | Create legibility filter module | 1 | High |
| bbva-0bw-2 | Create SmolVLM2 OCR service | 1 | High |
| bbva-0bw-3 | Create jersey_number database model | 1 | High |
| bbva-0bw-4 | Integrate OCR with detection pipeline | 2 | High |
| bbva-0bw-5 | Create OCR aggregation service | 2 | Medium |
| bbva-0bw-6 | Implement track-to-player mapping | 3 | Medium |
| bbva-0bw-7 | Add API endpoints | 4 | Medium |
| bbva-0bw-8 | Write integration tests | 4 | Medium |

---

## Files to Create/Modify

| File | Action | Task |
|------|--------|------|
| `backend/app/ml/legibility_filter.py` | Create | 1.1 |
| `backend/app/ml/jersey_ocr.py` | Create | 1.2 |
| `backend/app/models/jersey_number.py` | Create | 1.3 |
| `backend/alembic/versions/*_add_jersey_numbers.py` | Create | 1.3 |
| `backend/app/services/detection_pipeline.py` | Modify | 2.1 |
| `backend/app/services/jersey_aggregator.py` | Create | 2.2 |
| `backend/app/services/player_identifier.py` | Create | 3.1 |
| `backend/app/api/detection.py` | Modify | 4.1 |
| `backend/tests/test_legibility_filter.py` | Create | 1.1 |
| `backend/tests/test_jersey_ocr.py` | Create | 1.2 |
| `backend/tests/test_jersey_aggregator.py` | Create | 2.2 |

---

## Dependencies to Add

```toml
# backend/pyproject.toml
transformers = ">=4.49.0"  # SmolVLM2 support
accelerate = "^0.27.0"     # Model loading
```

For Apple Silicon MLX optimization (optional):
```bash
pip install mlx-vlm
```

---

## Acceptance Criteria

- [ ] Legibility filter reduces OCR calls by 50%+ on blurry/small crops
- [ ] SmolVLM2 correctly reads jersey numbers from clear crops
- [ ] Results aggregated across frames per tracking_id
- [ ] Invalid numbers (>99, non-numeric) rejected
- [ ] No blocking of API during OCR processing
- [ ] At least 70% accuracy on visible jersey numbers
- [ ] Swap detection logs when track shows different numbers

---

## Research Findings (Detailed)

### SmolVLM2

**What it is**: Multimodal vision-language model from Hugging Face (256M - 2.2B params)

**Pros**:
- ✅ Native MLX support for Apple Silicon
- ✅ Video understanding built-in
- ✅ Fine-tunes well with ~80 jersey images
- ✅ Apache 2.0 license

**Cons**:
- ❌ Base model needs prompting for number extraction
- ❌ Slower than traditional OCR (~8-10s per frame)
- ❌ Fine-tuning improves accuracy significantly

**Usage**:
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
# Prompt: "What jersey number is shown in this image? Reply with just the number."
```

### ResNet

**What it is**: CNN classification treating each jersey number (0-99) as a class

**Pros**:
- ✅ 93% accuracy on basketball (Roboflow 2025)
- ✅ Very fast inference (<10ms)
- ✅ Well-understood architecture

**Cons**:
- ❌ Requires thousands of labeled training images
- ❌ No pre-trained jersey number model exists
- ❌ MPS support is problematic

**Decision**: Skip for now, add in Phase 2 if needed.

### PaddleOCR

**Status**: ❌ Not viable

- Severe compatibility issues on Apple Silicon (freezes)
- Struggles with small text
- No easy digit-only mode

### EasyOCR

**Status**: ⚠️ Backup option

- Works on CPU but struggles with single digits
- Could be used for quick prototyping
- Lower accuracy than SmolVLM2

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SmolVLM2 too slow | Run OCR only on sampled frames, cache results |
| Poor accuracy on blurry text | Legibility filter skips low-quality crops |
| MPS not working | Fall back to CPU, use MLX for Apple Silicon |
| Model too large | Start with 500M variant, upgrade if needed |
