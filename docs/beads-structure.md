# Beads Issue Structure

## Overview

This document outlines the beads issue structure for the Basketball Video Analyzer project. Issues are organized into epics (phases) with clear dependencies to enable parallel work where possible.

## Quick Reference

```bash
bd ready              # Show issues ready to work on (no blockers)
bd list --status open # List all open issues
bd show <issue-id>    # Show detailed issue information
bd dep tree <issue-id> # Visualize dependency tree
```

## Epic Structure

### Epic: Project Setup & Infrastructure (bbva-ey3)
**Priority**: P0
**Status**: Ready to work
**Description**: Initialize project structure, documentation, and development environment

**Child Tasks**:
- `bbva-78a` - Initialize Python backend with FastAPI and Poetry [P0] ✅ READY
- `bbva-8xn` - Initialize React frontend with Vite and TypeScript [P0] ✅ READY
- `bbva-alb` - Set up git repository with .gitignore [P0] ✅ READY
- `bbva-fkw` - Create project README and setup documentation [P0] ✅ READY

**Parallelization**: All 4 setup tasks can run in parallel

---

### Epic: Phase 1 - MVP Foundation (bbva-ifo)
**Priority**: P0
**Status**: Blocked (waiting on setup)
**Description**: Video management, unified game player, manual annotation, and player roster management

#### Database Layer (Sequential)
1. `bbva-isa` - Design and implement SQLAlchemy models [P0]
   - Blocks on: `bbva-78a` (backend init)
2. `bbva-fft` - Set up Alembic and create initial migration [P0]
   - Blocks on: `bbva-isa`
3. `bbva-45w` - Create seed data for testing [P1]
   - Blocks on: `bbva-fft`

#### Backend Services (Parallel after database)
All services block on `bbva-fft` (migrations), then can run in parallel:
- `bbva-8mv` - Implement video upload and storage service [P0]
- `bbva-03i` - Implement video metadata extraction service [P0]
- `bbva-y0h` - Implement game timeline sequencing service [P0]
- `bbva-ryz` - Implement video thumbnail generation service [P1]

#### Backend API Endpoints (Parallel after services)
- `bbva-bc9` - Implement Games CRUD API endpoints [P0]
  - Blocks on: `bbva-8mv`
- `bbva-sl9` - Implement Videos CRUD API endpoints [P0]
  - Blocks on: `bbva-8mv`, `bbva-y0h`
- `bbva-788` - Implement Players CRUD API endpoints [P0]
  - Blocks on: `bbva-isa`
- `bbva-x6c` - Implement Game Rosters API endpoints [P0]
  - Blocks on: `bbva-788`
- `bbva-56k` - Implement Annotations CRUD API endpoints [P0]
  - Blocks on: `bbva-y0h`

#### Frontend Core (Parallel after frontend init)
State Management:
- `bbva-rx5` - Implement unified game timeline state management [P0]
  - Blocks on: `bbva-8xn` (frontend init)

Components (all block on state management):
- `bbva-wi7` - Implement multi-video playback engine [P0]
  - Blocks on: `bbva-rx5`
- `bbva-6c1` - Implement game timeline player UI component [P0]
  - Blocks on: `bbva-wi7`
- `bbva-jb6` - Implement timeline visualization component [P1]
  - Blocks on: `bbva-rx5`
- `bbva-4xa` - Implement video sequencing UI [P1]
  - Blocks on: `bbva-rx5`
- `bbva-4m7` - Implement annotation interface [P0]
  - Blocks on: `bbva-rx5`
- `bbva-f83` - Implement player/roster management UI [P1]
  - Blocks on: `bbva-rx5`

API Integration:
- `bbva-t11` - Implement API client service [P0]
  - Blocks on: `bbva-8xn`

#### Pages (After components + API)
- `bbva-d4b` - Implement Game List page [P1]
  - Blocks on: `bbva-t11`, `bbva-bc9`
- `bbva-0t9` - Implement Game Detail page [P1]
  - Blocks on: `bbva-t11`, `bbva-sl9`, `bbva-f83`
- `bbva-0ty` - Implement Video Analysis page [P0]
  - Blocks on: `bbva-6c1`, `bbva-4m7`

#### Testing & Polish
- `bbva-7k4` - Write backend unit tests [P1]
  - Blocks on: `bbva-56k`
- `bbva-qqg` - Write frontend component tests [P2]
  - Blocks on: `bbva-0ty`
- `bbva-1fw` - End-to-end testing with real video [P1]
  - Blocks on: `bbva-0ty`
- `bbva-s5h` - Performance optimization for large videos [P2]
  - Blocks on: `bbva-1fw`
- `bbva-bw0` - Create user documentation for MVP [P2]

---

### Epic: Phase 2 - Computer Vision Integration (bbva-g2m)
**Priority**: P1
**Status**: Ready to work
**Description**: Semi-automatic player detection, tracking, and jersey number recognition

#### ML Foundation (Sequential Pipeline)
1. `bbva-fut` - Set up ML model infrastructure (YOLOv8-nano, PyTorch) [P0] ✅ READY
2. `bbva-olg` - Implement video frame extraction service [P0]
   - Blocks on: `bbva-fut`
3. `bbva-tlh` - Implement player detection service with YOLOv8 [P0]
   - Blocks on: `bbva-fut`, `bbva-olg`
4. `bbva-yjv` - Implement player tracking service with ByteTrack [P0]
   - Blocks on: `bbva-tlh`

#### Background Processing (Parallel with ML Foundation)
5. `bbva-12x` - Implement background processing job system [P0] ✅ READY
6. `bbva-r0t` - Create player detection API endpoints [P1]
   - Blocks on: `bbva-yjv`, `bbva-12x`

#### Player Identification
7. `bbva-0oh` - Implement jersey number OCR service [P1]
   - Blocks on: `bbva-tlh`
8. `bbva-59j` - Implement player-detection matching service [P1]
   - Blocks on: `bbva-0oh`, `bbva-yjv`

#### Frontend Detection UI
9. `bbva-90h` - Implement detection overlay component [P1]
   - Blocks on: `bbva-r0t`
10. `bbva-qjj` - Implement detection review UI [P1]
    - Blocks on: `bbva-90h`, `bbva-59j`
11. `bbva-99a` - Implement processing status UI [P1]
    - Blocks on: `bbva-12x`

#### Optional Enhancements
12. `bbva-e5q` - Implement court detection service [P2]
    - Blocks on: `bbva-fut`, `bbva-olg`

#### Testing
13. `bbva-x0n` - Write ML pipeline unit tests [P1]
    - Blocks on: `bbva-yjv`, `bbva-0oh`
14. `bbva-kz5` - End-to-end testing with CV pipeline [P1]
    - Blocks on: `bbva-qjj`, `bbva-99a`

**Parallelization Opportunities**:
- `bbva-fut` (ML Infrastructure) and `bbva-12x` (Background Jobs) can run in parallel
- After ML pipeline is complete, OCR and Detection API can run in parallel
- Frontend components (overlay, status UI) can run in parallel after their backend deps

---

### Epic: Phase 3 - Play Recognition (bbva-sak)
**Priority**: P1
**Status**: Ready (can plan in parallel)
**Description**: Automatic play detection and categorization with user verification

**Note**: Tasks not yet created. Will be broken down when Phase 2 is underway.

---

### Epic: Phase 4 - Search & Export (bbva-7d0)
**Priority**: P2
**Status**: Ready (can plan in parallel)
**Description**: Advanced search, filtering, clip generation, and highlight reels

**Note**: Tasks not yet created. Will be broken down when Phase 3 is underway.

---

### Epic: Phase 5 - Advanced Multi-Angle Features (bbva-qbe)
**Priority**: P2
**Status**: Ready (can plan in parallel)
**Description**: Enhanced multi-angle viewing and automatic video synchronization

**Note**: Tasks not yet created. Will be broken down when Phase 4 is underway.

---

### Epic: Phase 6 - Desktop App (bbva-yoy)
**Priority**: P3
**Status**: Ready (can plan in parallel)
**Description**: Electron packaging and native desktop distribution

**Note**: Tasks not yet created. Will be broken down when Phase 1 MVP is stable.

---

## Parallelization Opportunities

### Immediate (Now)
All setup tasks can run in parallel:
- Backend initialization
- Frontend initialization
- Git repository setup
- Documentation

### Phase 1 - Wave 1 (After backend init)
- Database models design (single task, not parallelizable)

### Phase 1 - Wave 2 (After migrations)
All backend services can run in parallel:
- Video upload service
- Metadata extraction service
- Timeline sequencing service
- Thumbnail generation service

### Phase 1 - Wave 3 (After services)
Backend and Frontend can run **fully in parallel**:

**Backend track**:
- Games API
- Videos API
- Players API
- Rosters API (after Players)
- Annotations API

**Frontend track** (after frontend init):
- State management
- Then 6 components in parallel
- API client service

### Phase 1 - Wave 4 (Integration)
- Pages (some dependencies between them)
- Testing

## Dependency Visualization

To see the dependency tree for any issue:
```bash
bd dep tree bbva-ifo  # See all Phase 1 dependencies
bd dep tree bbva-8mv  # See what blocks video upload service
```

## Working on Tasks

### Workflow
1. Check ready work: `bd ready`
2. Pick a task: `bd show <issue-id>`
3. Create feature branch: `git checkout -b feature/<issue-id>-short-desc`
4. Update status: `bd update <issue-id> --status in_progress`
5. Do the work, commit with conventional commits including issue ID
6. Create PR when complete
7. Close issue: `bd close <issue-id> --reason "Completed in PR #X"`

### Example Commands
```bash
# Start work on backend initialization
bd show bbva-78a
git checkout -b feature/bbva-78a-backend-init
bd update bbva-78a --status in_progress

# Make changes, commit
git add .
git commit -m "feat(backend): [bbva-78a] initialize FastAPI project with Poetry"
git push origin feature/bbva-78a-backend-init

# After PR merged
bd close bbva-78a --reason "Completed in PR #1"
```

## Next Steps

1. **Phase 1 Complete** ✅
   - All 37 Phase 1 tasks closed
   - Backend API fully functional
   - Frontend MVP with multi-video timeline
   - Manual annotation workflow working

2. **Phase 2: Computer Vision** (Current)
   - Start with `bbva-fut` (ML Infrastructure) and `bbva-12x` (Background Jobs) in parallel
   - Then sequential ML pipeline: Frame Extraction → Detection → Tracking
   - Parallel tracks: OCR and Detection API after dependencies clear
   - Frontend UI after backend APIs ready

3. **Future phases**: Break down when ready to start

## Issue Naming Convention

Issues follow the format: `bbva-<random>`
- `bbva` = basketball-video-analyzer
- Random suffix for uniqueness

## Commit Message Format

All commits must follow:
```
<type>(<scope>): [<issue-id>] <description>

Examples:
feat(backend): [bbva-78a] initialize FastAPI with Poetry
fix(frontend): [bbva-6c1] fix video player timeline position
docs: [bbva-fkw] add installation instructions to README
test(backend): [bbva-7k4] add unit tests for video service
```

## Current Status Summary

- **Total Issues**: 56
- **Phase 1**: Complete (37 closed)
- **Phase 2**: 14 tasks created
- **Future Phases**: 4 epics (Phase 3-6)
- **Ready to Work**: 7 issues (2 P0 Phase 2 tasks + 5 future epics)
- **Blocked**: 12 tasks (waiting on dependencies)

**Recommended Starting Point**: Work on `bbva-fut` (ML Infrastructure) and `bbva-12x` (Background Jobs) in parallel.
