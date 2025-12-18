# CLAUDE.md - Optimized for OpenCode

**Basketball Video Analyzer** - Local-first youth basketball game analysis tool.

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Backend** | Python 3.11+, FastAPI, SQLAlchemy, SQLite |
| **Frontend** | React 18+, TypeScript, Mantine UI, Zustand |
| **ML** | YOLOv8-nano for player detection |
| **Issue Tracker** | Beads (`bd` CLI) |
| **Commits** | `type(scope): [bbva-xxx] description` |

## Critical Architecture Pattern

**Multi-Video Timeline**: Multiple camera angles stitched into unified game timeline.
- Annotations reference `game_timestamp` (unified), not individual video timestamps
- Cross-video annotations via `annotation_videos` junction table
- Timeline calculation in `backend/app/services/timeline_sequencer.py`

## Key Database Relationships

```
games → videos (one-to-many)
games → game_rosters ← players (many-to-many)  
annotations ← annotation_videos → videos (many-to-many)
annotations → plays (one-to-one)
videos → player_detections (one-to-many)
```

## Development Workflow

1. Check work: `bd ready`
2. Claim issue: `bd update <id> --status=in_progress` 
3. Make changes, test
4. Commit: `git commit -m "type(scope): [bbva-xxx] description"`
5. Push: `git push origin main`
6. Close: `bd close <id> --reason="description"`

## Code Organization

### Backend Structure
```
backend/app/
├── models/         # SQLAlchemy ORM models
├── schemas/        # Pydantic validation  
├── api/           # FastAPI route handlers (thin)
├── services/      # Business logic
├── ml/            # ML model inference
└── main.py        # FastAPI app
```

### Frontend Structure  
```
frontend/src/
├── components/    # React components
├── pages/         # Page-level components
├── store/         # Zustand state management
├── api/           # Backend API client
└── types/         # TypeScript definitions
```

## Important Patterns

- **Video Processing**: Async, FFmpeg for metadata, OpenCV for processing
- **Annotations**: Use unified `game_timestamp`, store video-specific segments in junction table
- **State Management**: FastAPI dependency injection, Zustand for global state
- **Frontend**: Delegate ALL .tsx/.jsx/.css files to frontend-ui-ux-engineer agent

## Current Phase

**Phase 2: Computer Vision** - Player detection, tracking, jersey number recognition.

## Context Optimization

This project uses `.opencodeignore` to exclude large files from context:
- Video files (backend/videos/)
- Dependencies (node_modules/, venv/)
- Build artifacts (dist/, build/)
- Database files (*.db)
- Lock files (pnpm-lock.yaml, poetry.lock)

This prevents quota exhaustion and "prompt too large" issues.

See docs/ for detailed implementation plans when needed.