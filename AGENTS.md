# AI Agent Instructions

**All AI agents (Claude, Copilot, etc.) must read this file first.** This is the single source of truth for development context.

## Required Reading Before Starting

1. **[docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)** - Git workflow, commit conventions, coding standards
2. **[docs/implementation-plan.md](./docs/implementation-plan.md)** - Complete technical specs (read on-demand for specific features)
3. **[docs/beads-structure.md](./docs/beads-structure.md)** - Task organization (read on-demand)

## Project Summary

Basketball Video Analyzer - local-first app for analyzing youth basketball games. Parents/coaches can analyze full games in <1 hour, find player plays in <2 minutes, generate highlight reels in <5 minutes.

**Current Phase**: Phase 2 - Computer Vision (ML infrastructure complete)

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18+, TypeScript, Mantine UI, Zustand, Vite |
| Backend | Python 3.11+, FastAPI, SQLAlchemy, SQLite |
| ML/Video | YOLOv8-nano, ByteTrack, OpenCV, FFmpeg |
| Issue Tracker | Beads (`bd` CLI) - **NOT GitHub Issues** |
| Git | Trunk-based - commit directly to `main` |

## Database Schema (Key Relationships)

```
games → videos (1:many)
games ↔ players via game_rosters (many:many)
annotations ↔ videos via annotation_videos (many:many)
annotations → plays (1:1)
videos → player_detections (1:many)
```

## Development Workflow

**IMPORTANT**: We have been having prompt too large errors when interacting - this is a known bug.  When this happens, we have to start a new fresh session.  To aid in recovery, keep a summary of what we recently accomplished, we're working on, and what the next steps will be in a task-summy document located in `docs/tasks/<id>-summary.md`.

```bash
# 1. Check for work
bd ready

# 2. Claim issue
bd update <id> --status=in_progress

# 3. Document our plan in `docs/tasks/<id>.md`

# 4. Make changes, update the plan-summary doc, test locally

# 5. update our plan document with implementation details as a seperate section - leave the plan intact.

# 6. Commit (trunk-based - direct to main)
git commit -m "type(scope): [bbva-xxx] description"
git push origin main

# 7. Close issue
bd close <id> --reason="description"

```

## Key Patterns

**Video Processing**
- Process asynchronously (long-running)
- FFmpeg for metadata → OpenCV for processing
- Store metadata in DB, generate thumbnails on upload

**Annotations**
- Reference `game_timestamp`, not video timestamps
- Use `annotation_videos` for cross-video segments
- Support "unverified" ML annotations with confidence scores

**State Management**
- Backend: FastAPI dependency injection
- Frontend: Zustand for global state, local for video player

## Code Organization

Backend: `models/` (ORM), `schemas/` (validation), `api/` (routes), `services/` (logic), `ml/` (inference)

Frontend: `components/`, `pages/`, `store/`, `api/`, `types/`

## When You Need More Context

- Multi-video timeline details → `docs/implementation-plan.md` section "Unified Game Timeline"
- Task dependencies → `docs/beads-structure.md`
- Full architecture → `docs/implementation-plan.md`
- Git conventions → `docs/CONTRIBUTING.md`

**Default**: Start work with this file only. Read additional docs when needed for specific features.
