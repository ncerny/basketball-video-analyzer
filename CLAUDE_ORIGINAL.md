# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Development Guidelines**: See **[docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)** for git workflow, commit conventions, and coding standards.

## Project Overview

Basketball Video Analyzer is a local-first, multi-platform application for analyzing youth basketball game videos with semi-automatic player tracking and play tagging. The project enables parents and coaches to analyze full games in under 1 hour, find all plays by a specific player in under 2 minutes, and generate highlight reels in under 5 minutes.

**Current Status**: Active development - Phase 1 MVP and Phase 2 CV integration in progress.

## Architecture

The system uses a layered architecture:

- **Frontend Layer**: React 18+ with TypeScript, Vite, Zustand state management, Tailwind CSS
- **Backend Layer**: Python 3.11+ FastAPI server for video management, annotation CRUD, and ML orchestration
- **Processing Layer**: OpenCV/FFmpeg for video processing, YOLOv8-nano for player detection, ByteTrack for tracking
- **Storage Layer**: SQLite database with local filesystem storage for videos

### Critical Architectural Pattern: Unified Game Timeline

The most complex aspect of this system is **stitching multiple videos from different cameras into a seamless, continuous game timeline**. This requires:

1. **Video Timeline Calculation**: Extract `recorded_at` timestamp from video metadata, calculate `sequence_order` and `game_time_offset` for each video
2. **Game Time ↔ Video Time Conversion**: Convert between unified game timeline and individual video timestamps
3. **Cross-Video Annotations**: A single play annotation can span multiple videos via the `annotation_videos` junction table
4. **Overlap Handling**: Detect and manage overlapping videos from multiple camera angles

See docs/implementation-plan.md section "Unified Game Timeline with Multiple Videos" for implementation details.

## Database Schema

Key tables and their relationships:

- `games` → `videos` (one-to-many): Each game has multiple videos from different cameras
- `games` → `game_rosters` ← `players` (many-to-many): Players assigned to teams per game
- `annotations` ← `annotation_videos` → `videos` (many-to-many): Annotations can span multiple videos
- `annotations` → `plays` (one-to-one): Detailed play information
- `videos` → `player_detections` (one-to-many): ML detection results per frame

Critical fields:
- `videos.recorded_at`: Timestamp when recording started (for timeline calculation)
- `videos.sequence_order`: Order of videos in game timeline
- `videos.game_time_offset`: Offset from game start (in seconds)
- `annotation_videos.video_timestamp_start/end`: Video-specific timestamps for the annotation segment

## Development Workflow

See **[docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)** for complete development guidelines including:
- Git workflow (trunk-based development - commit directly to `main`)
- Commit conventions
- Development commands (backend/frontend)
- Issue management with Beads
- Code standards and testing

## Key Technical Challenges

### 1. Multi-Video Timeline Synchronization
- **Problem**: Seamlessly stitch multiple videos from different cameras into continuous playback
- **Solution**: Calculate game time offsets from video metadata, support cross-video annotations
- **Implementation**: See `backend/app/services/timeline_sequencer.py` (future)

### 2. Local ML Performance
- **Problem**: Running computer vision models on consumer hardware
- **Solution**: Use lightweight models (YOLOv8-nano ~6MB), frame sampling, background processing
- **Implementation**: See `backend/app/ml/yolo_detector.py` (future)

### 3. Player Re-identification Across Videos
- **Problem**: Tracking the same player across different videos/cameras
- **Solution**: Jersey numbers as primary ID, appearance features for confirmation, manual override
- **Implementation**: See `backend/app/services/tracker.py` (future)

### 4. Play Boundary Detection
- **Problem**: Determining exact start/end timestamps of plays
- **Solution**: Generous buffers (±10 seconds), user trimming interface, learn from corrections
- **Implementation**: See `backend/app/services/play_recognizer.py` (future)

## Code Organization Principles

### Backend Structure (planned)
```
backend/app/
├── models/        # SQLAlchemy ORM models (database tables)
├── schemas/       # Pydantic models (API request/response validation)
├── api/           # FastAPI route handlers (thin layer, delegate to services)
├── services/      # Business logic (video processing, timeline calculation, etc.)
├── ml/            # ML model inference (YOLO, tracking, OCR)
└── main.py        # FastAPI app initialization
```

### Frontend Structure (planned)
```
frontend/src/
├── components/    # Reusable React components
│   ├── VideoPlayer/     # Multi-video synchronized playback
│   ├── Timeline/        # Game timeline visualization
│   ├── AnnotationForm/  # Play annotation interface
│   └── ...
├── pages/         # Page-level components (GameList, GameDetail, VideoAnalysis)
├── store/         # Zustand state management stores
├── api/           # Backend API client (fetch/axios)
└── types/         # TypeScript type definitions
```



## Important Patterns

### Video Processing
- Always process videos asynchronously (long-running operations)
- Use FFmpeg for metadata extraction before OpenCV processing
- Store video metadata (duration, fps, resolution, recorded_at) in database
- Generate thumbnails at upload time for UI preview

### Annotation System
- Annotations reference `game_timestamp` (unified timeline), not individual video timestamps
- Use `annotation_videos` junction table to map annotations to specific video segments
- Support "unverified" annotations from ML with confidence scores
- Allow users to verify/edit/reject ML-generated annotations

### State Management
- Backend: Use FastAPI dependency injection for database sessions
- Frontend: Use Zustand stores for global state (games, videos, annotations)
- Keep video player state local to component
- Debounce timeline scrubbing for performance

## Project Phases

The project is organized into 6 phases tracked as epics in Beads:

1. **Phase 1: MVP Foundation** ✅ - Video management, manual annotation, player management, basic timeline
2. **Phase 2: Computer Vision** 🔄 - Player detection, tracking, jersey number recognition
3. **Phase 3: Play Recognition** - Automatic play detection and classification
4. **Phase 4: Search & Export** - Advanced filtering, clip generation, highlight reels
5. **Phase 5: Multi-Angle Features** - Enhanced multi-angle viewing and synchronization
6. **Phase 6: Desktop App** - Electron packaging and distribution

**Current Phase**: Phase 2 - Computer Vision (ML infrastructure complete)

See docs/beads-structure.md for complete task breakdown and dependencies.



## Documentation

- `docs/implementation-plan.md`: Comprehensive technical architecture and specifications (671 lines)
- `docs/beads-structure.md`: Issue tracker organization and task dependencies (271 lines)

All documentation is current and detailed. Read these files before starting implementation work.
