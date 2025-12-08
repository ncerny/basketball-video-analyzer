# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Basketball Video Analyzer is a local-first, multi-platform application for analyzing youth basketball game videos with semi-automatic player tracking and play tagging. The project enables parents and coaches to analyze full games in under 1 hour, find all plays by a specific player in under 2 minutes, and generate highlight reels in under 5 minutes.

**Current Status**: Planning phase complete with comprehensive documentation. Backend and frontend implementation not yet started.

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

## Development Commands

### Backend (not yet initialized)
```bash
cd backend
poetry install                              # Install dependencies
poetry run pytest                           # Run tests
poetry run pytest tests/test_specific.py    # Run single test file
poetry run uvicorn app.main:app --reload   # Start dev server on http://localhost:8000
poetry run alembic upgrade head             # Apply database migrations
poetry run alembic revision --autogenerate -m "description"  # Create new migration
```

### Frontend (not yet initialized)
```bash
cd frontend
pnpm install          # Install dependencies
pnpm dev              # Start dev server on http://localhost:5173
pnpm build            # Build for production
pnpm test             # Run all tests
pnpm test -- MyComponent  # Run tests for specific component
pnpm lint             # Run ESLint
```

### Issue Management (Beads)

This project uses **Beads issue tracker** instead of GitHub Issues. Run `bd prime` at session start for workflow context.

**Core Principles:**
- Track ALL work in beads (no TodoWrite tool, no markdown TODOs)
- Use `bd create` to create issues, NOT TodoWrite tool
- Git workflow: hooks auto-sync, run `bd sync` at session end
- Session management: check `bd ready` for available work

**Essential Commands:**

**Finding Work:**
```bash
bd ready                     # Show issues ready to work (no blockers)
bd list --status=open        # All open issues
bd list --status=in_progress # Your active work
bd show <id>                 # Detailed issue view with dependencies
bd blocked                   # Show all blocked issues
```

**Creating & Updating:**
```bash
bd create --title="..." --type=task|bug|feature|epic|chore
bd update <id> --status=in_progress     # Claim work
bd update <id> --assignee=username      # Assign to someone
bd close <id>                           # Mark complete
bd close <id1> <id2> ...                # Close multiple (more efficient)
bd close <id> --reason="explanation"    # Close with reason
bd reopen <id>                          # Reopen closed issue
```

**Dependencies & Blocking:**
```bash
bd dep add <issue> <depends-on>  # Add dependency (issue depends on depends-on)
bd show <id>                     # See what's blocking/blocked by this issue
```

**Sync & Project Health:**
```bash
bd sync                # Sync with git remote (run at session end)
bd sync --status       # Check sync status without syncing
bd stats               # Project statistics (open/closed/blocked counts)
```

**Common Workflows:**

Starting work:
```bash
bd ready                              # Find available work
bd show <id>                          # Review issue details
bd update <id> --status=in_progress   # Claim it
```

Completing work:
```bash
bd close <id1> <id2> ...    # Close all completed issues at once
bd sync                     # Push beads changes to remote
```

Creating dependent work:
```bash
bd create --title="Implement feature X" --type=feature
bd create --title="Write tests for X" --type=task
bd dep add <test-id> <feature-id>  # Tests depend on Feature
```

**MCP Server Tools (Alternative):**

For use within code/automation, use MCP tools:
- `mcp__plugin_beads_beads__ready`: Show ready-to-work issues
- `mcp__plugin_beads_beads__list`: List issues with filters
- `mcp__plugin_beads_beads__show`: Show detailed issue info
- `mcp__plugin_beads_beads__create`: Create new issue
- `mcp__plugin_beads_beads__update`: Update issue status/priority
- `mcp__plugin_beads_beads__close`: Close completed issue
- `mcp__plugin_beads_beads__dep`: Add/manage dependencies

**Issue Types**: Epic, Feature, Task, Bug, Chore
**Issue ID Format**: `bbva-<random>` (e.g., `bbva-a58df2`)
**Dependencies**: `blocks`, `related`, `parent-child`, `discovered-from`

## Git Workflow

### Branch Strategy
Scaled Trunk-Based Development:
- `main` branch is always deployable
- Short-lived feature branches (1-2 days max)
- Feature branches named: `feature/<issue-id>-short-description`
- Releases tagged from `main` using semantic versioning (e.g., `v1.0.0`)

### Commit Convention
Every commit MUST follow Conventional Commit syntax with the beads issue ID:

**Format**: `<type>(<scope>): [<issue-id>] <description>`

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples**:
```bash
git commit -m "feat(backend): [bbva-a58df2] add video upload endpoint"
git commit -m "fix(frontend): [bbva-b72f1a] fix timeline scrubber position"
git commit -m "test(backend): [bbva-a58df2] add unit tests for video processor"
```

### Workflow Steps

**🚨 CRITICAL: NEVER push directly to main branch! Always create a feature branch and PR.**

**1. Start Work**
- Pull latest: `git pull origin main`
- Get ready issues: `bd ready` or use MCP tool `mcp__plugin_beads_beads__ready()`
- Show issue details: `bd show <issue-id>`
- Create feature branch: `git checkout -b feature/<issue-id>-short-description`
- Claim work: `bd update <issue-id> --status=in_progress`

**2. Make Changes**
- Write code in small, logical chunks
- Write tests for new functionality
- Ensure all tests pass before committing

**3. Commit Changes (Code Only)**
- Stage ONLY code files: `git add <code-files>`
  - ⚠️ **NEVER stage `.beads/` directory** - beads manages this separately
  - Example: `git add frontend/src/pages/GameDetail.tsx`
- Commit code with Conventional Commit format:
  ```bash
  git commit -m "feat(scope): [bbva-xxx] description"
  ```
- Format: `<type>(<scope>): [<issue-id>] <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`
- Examples:
  - `git commit -m "feat(backend): [bbva-a58df2] add video upload endpoint"`
  - `git commit -m "fix(frontend): [bbva-b72f1a] fix timeline scrubber position"`

**4. Sync Beads Database (Separate from Code)**
- Run `bd sync` to push beads changes directly to main
- This happens INDEPENDENTLY of your feature branch
- Beads changes are NOT included in your PR
- `bd sync` manages issue status, updates, closures, etc.

**5. Push Feature Branch**
- Use Bash tool: `git push origin feature/<issue-id>-description`
- **NEVER use `git push origin main`** - this pushes directly to main which is forbidden
- Your feature branch should contain ONLY code changes, NOT beads database changes

**6. Create Pull Request**
- Use `mcp__github__create_pull_request` tool with:
  - `head`: feature branch name (e.g., `feature/bbva-xxx-description`)
  - `base`: `main`
  - `title`: `[bbva-xxx] Description`
  - `body`: Summarize changes and reference beads issue
- Example:
  ```bash
  mcp__github__create_pull_request(
    owner="ncerny",
    repo="basketball-video-analyzer",
    title="[bbva-f83] Implement player management UI",
    head="feature/bbva-f83-player-management",
    base="main",
    body="Implements full CRUD player management..."
  )
  ```

**7. After PR Merge**
- Close issue: `bd close <issue-id> --reason="Completed in PR #42"`
- Or use MCP: `mcp__plugin_beads_beads__close(issue_id="bbva-xyz", reason="Completed in PR #42")`
- Run `bd sync` to push the closure to main
- Delete feature branch locally: `git branch -d feature/<issue-id>-description`
- GitHub will automatically delete the remote branch if configured

**8. Session End Checklist**

Before ending any work session, run this checklist:
```bash
bd sync                 # Sync beads database with git
git status              # Verify no uncommitted changes
git push origin <branch>  # Push feature branch (NOT main!)
```

### Semantic Versioning

Releases follow [Semantic Versioning](https://semver.org/): **MAJOR.MINOR.PATCH**

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

**Release Process:**
1. Ensure `main` is stable and all tests pass
2. Update version in `package.json` and `pyproject.toml`
3. Create release commit: `chore: [bbva-release] bump version to v1.2.0`
4. Use Bash tool to tag: `git tag -a v1.2.0 -m "Release v1.2.0"`
5. Push with tags: `git push origin main --tags`
6. Optionally create GitHub Release using `mcp__github__*` tools

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

## Testing Strategy

### Backend
- Use pytest with fixtures for database setup
- Test services independently with mocked dependencies
- Integration tests for API endpoints
- Mock ML models in tests (don't run actual inference)

### Frontend
- Component tests with Vitest + React Testing Library
- Test user interactions and state changes
- Mock API calls with MSW (Mock Service Worker)
- Snapshot tests for complex UI components

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

1. **Phase 1: MVP Foundation** - Video management, manual annotation, player management, basic timeline
2. **Phase 2: Computer Vision** - Player detection, tracking, jersey number recognition
3. **Phase 3: Play Recognition** - Automatic play detection and classification
4. **Phase 4: Search & Export** - Advanced filtering, clip generation, highlight reels
5. **Phase 5: Multi-Angle Features** - Enhanced multi-angle viewing and synchronization
6. **Phase 6: Desktop App** - Electron packaging and distribution

**Current Phase**: Pre-Phase 1 (setup and initialization)

See docs/beads-structure.md for complete task breakdown and dependencies.

## Best Practices

### Code Organization
- Keep components small and focused (Single Responsibility Principle)
- Use meaningful names for files, functions, and variables
- Write self-documenting code with clear intent
- Add comments only when "why" isn't obvious from code

### Testing
- Write tests before or alongside code (TDD encouraged)
- Aim for high coverage of critical paths
- Test edge cases and error conditions
- Keep tests fast and isolated

### Git Hygiene
- Write clear, descriptive commit messages
- Keep commits atomic (one logical change per commit)
- Don't commit commented-out code
- Don't commit debugging statements (console.log, print, etc.)
- Review your own diff before committing

### Pull Requests
- Keep PRs small (< 400 lines changed when possible)
- Write clear PR descriptions explaining what and why
- Include screenshots/videos for UI changes
- Link to related beads issues
- Respond to review comments promptly

### Performance
- Profile before optimizing
- Optimize for readability first, performance second
- Document any performance-critical code
- Use lazy loading for large video files

### Security
- Never commit secrets, API keys, or credentials
- Sanitize user input
- Validate data at API boundaries
- Use parameterized queries (SQLAlchemy ORM handles this)

## GitHub MCP Server Tools

Use GitHub MCP tools for all GitHub operations:

**Repository & Branch Management:**
- `mcp__github__list_branches`: List all branches
- `mcp__github__create_branch`: Create new branch from base

**Pull Requests:**
- `mcp__github__create_pull_request`: Create new PR
- `mcp__github__list_pull_requests`: List PRs with filters
- `mcp__github__pull_request_read`: Get PR details, diff, files, comments, reviews
- `mcp__github__update_pull_request`: Update PR title, body, reviewers
- `mcp__github__merge_pull_request`: Merge PR (squash/rebase/merge)
- `mcp__github__add_issue_comment`: Add comment to PR (as issue)
- `mcp__github__pull_request_review_write`: Create/submit PR review

**Commits:**
- `mcp__github__list_commits`: List commits for branch
- `mcp__github__get_commit`: Get commit details with diff

**Issues:**
- `mcp__github__issue_read`: Get issue details
- `mcp__github__issue_write`: Create or update issue
- `mcp__github__list_issues`: List issues with filters
- `mcp__github__add_issue_comment`: Add comment to issue

**Search:**
- `mcp__github__search_code`: Search code across repositories
- `mcp__github__search_issues`: Search issues
- `mcp__github__search_pull_requests`: Search PRs

## Documentation

- `docs/implementation-plan.md`: Comprehensive technical architecture and specifications (671 lines)
- `docs/beads-structure.md`: Issue tracker organization and task dependencies (271 lines)

All documentation is current and detailed. Read these files before starting implementation work.
