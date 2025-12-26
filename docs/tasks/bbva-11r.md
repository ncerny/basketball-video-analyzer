# bbva-11r: Enhance SAM2 Tracker with Embedding Memory

## Summary

Improve SAM2 tracking with feature embeddings for robust player re-identification across occlusions, video segments, and games.

## Implementation Plan

### Phase 1: Foundation

1. **Disable OCR** - Set `ENABLE_JERSEY_OCR=false` in config
2. **Auto-download SAM2 models** - Download missing checkpoints on first use
3. **Add new config settings** - Embedding thresholds, re-identification flags

### Phase 2: Embedding Extraction

4. **Extract embeddings from SAM2 encoder** - Access internal image encoder output
5. **Update TrackedObject dataclass** - Add embedding field
6. **Implement cosine similarity matching** - Replace/augment IOU matching

### Phase 3: Color Tiebreaker

7. **Integrate color histograms** - Use existing `color_extractor.py`
8. **Implement tiebreaker logic** - When embedding scores are close

### Phase 4: Database Persistence

9. **Create player_embeddings table** - Alembic migration
10. **Persist embeddings at video end** - Store best embedding per track
11. **Load embeddings at video start** - For cross-video matching

### Phase 5: Cross-Video Matching

12. **Query database for known embeddings** - When no in-memory match
13. **Link to player records** - Enable cross-game identity

## Files to Create

- `backend/app/models/player_embedding.py` - SQLAlchemy model
- `backend/scripts/download_sam2_model.py` - Auto-download script

## Files to Modify

- `backend/app/config.py` - New settings
- `backend/app/ml/sam2_tracker.py` - Core changes
- `backend/app/services/batch_processor.py` - Persist embeddings

## New Config Settings

```python
# Track persistence
sam2_lost_track_frames: int = 0  # 0 = keep tracks for entire video

# Embedding matching
sam2_embedding_similarity_threshold: float = 0.5
sam2_color_tiebreaker_threshold: float = 0.15

# Re-identification
sam2_reidentification_enabled: bool = True
sam2_auto_download: bool = True
```

## Future Enhancements (Not Implemented)

### Video Predictor Mode

SAM2's Video Predictor offers built-in memory and mask propagation:
- Requires frames saved to disk as JPEGs
- Designed for offline batch processing
- Could be added as post-processing refinement step
- Better for long occlusions and complex scenes

**Why deferred:** Our streaming architecture processes frames in memory. Video Predictor requires architectural changes that don't fit current design.

### Keyframe Masks

Store high-quality masks at key moments:
- Useful when embeddings are similar between players
- Higher memory cost (~100KB-1MB per mask)
- Could combine with embeddings for hybrid matching

**Why deferred:** Embeddings should handle most cases. Keyframes add complexity and memory. Can revisit if embedding-only approach is insufficient.

## Testing Checklist

- [ ] Unit test: Embedding extraction produces consistent vectors
- [ ] Unit test: Cosine similarity matching works correctly
- [ ] Unit test: Color tiebreaker activates when scores are close
- [ ] Unit test: Auto-download fetches missing models
- [ ] Integration test: Track maintains ID through brief occlusion
- [ ] Manual test: Run detection with player occlusions
- [ ] Manual test: Cross-video matching works

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Foundation | ✅ Complete | OCR disabled, auto-download added, config settings added |
| Phase 2: Embedding Extraction | ✅ Complete | Embeddings extracted from SAM2 encoder, EMA blending |
| Phase 3: Color Tiebreaker | ✅ Complete | Color histogram comparison when scores within threshold |
| Phase 4: Database Persistence | Not Started | |
| Phase 5: Cross-Video Matching | Not Started | |

## Bug Fixes Applied

1. **SAM2.1 vs SAM2 checkpoint compatibility** - SAM2.1 checkpoints (092824) incompatible with installed library. Switched to SAM2 (072824) URLs.

2. **OCR not respecting `enable_jersey_ocr` setting** - Fixed in multiple places:
   - `app/api/detection.py` - Added to job metadata
   - `app/services/batch_orchestrator.py` - Read from settings
   - `app/services/detection_pipeline.py` - Read from settings

3. **SAM2TrackerConfig defaults mismatch** - Synced dataclass defaults with global `sam2_*` settings.

4. **Tracks deleted too quickly** - Changed `lost_track_frames` from 30 to 0 (keep all tracks for entire video).

## Debug Logging Added

- SAM2 feature keys logged on first frame
- Embedding extraction count per frame
- Track matching decisions with scores and thresholds
- New track creation with reasoning
