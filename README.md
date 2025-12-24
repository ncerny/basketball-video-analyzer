# Basketball Video Analyzer

A local-first, multi-platform application for analyzing youth basketball game videos with semi-automatic player tracking and play tagging capabilities.

## Overview

Basketball Video Analyzer helps parents and coaches:
- **Analyze full games** in under 1 hour
- **Find all plays by a specific player** in under 2 minutes
- **Generate highlight reels** of a player's best plays in under 5 minutes

### Key Features

- **Multi-Video Timeline**: Seamlessly stitch multiple camera angles into a unified game timeline
- **Manual Annotation**: Tag plays with player information, play types, and timestamps
- **Player Management**: Build game rosters with jersey numbers and team assignments
- **Computer Vision** (future): Semi-automatic player detection and tracking
- **Play Recognition** (future): Automatic play detection and classification
- **Highlight Generation** (future): Create clip compilations of specific players or play types

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Frontend (React + TypeScript)                  │
│  Web App | Desktop (Electron - future)                  │
└─────────────────────────────────────────────────────────┘
                    ↓ REST API
┌─────────────────────────────────────────────────────────┐
│         Backend (Python FastAPI)                         │
│  - Video management                                      │
│  - Annotation CRUD                                       │
│  - ML model orchestration                                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│      Processing Layer (Computer Vision)                  │
│  - Video processing (OpenCV/FFmpeg)                      │
│  - Player detection (YOLO)                               │
│  - Player tracking (ByteTrack)                           │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│         Storage Layer (Local-First)                      │
│  - Video files (Local filesystem)                        │
│  - Metadata & Annotations (SQLite)                       │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI (async, fast, OpenAPI docs)
- **Database**: SQLite + SQLAlchemy ORM
- **Migrations**: Alembic
- **Video Processing**: OpenCV, FFmpeg
- **ML/CV**: YOLOv8-nano, ByteTrack, EasyOCR
- **Package Manager**: Poetry

### Frontend
- **Language**: TypeScript
- **Framework**: React 19+
- **Build Tool**: Vite 7
- **State Management**: Zustand
- **Styling**: Tailwind CSS 4
- **Routing**: React Router DOM 7
- **HTTP Client**: Axios
- **Testing**: Vitest + React Testing Library
- **Package Manager**: pnpm

### Development Tools
- **Git Workflow**: Scaled Trunk-Based Development
- **Issue Tracking**: Beads (custom issue tracker in `.beads/`)
- **Commit Convention**: Conventional Commits with issue IDs

## Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 18 or higher
- **Poetry**: For Python dependency management ([installation guide](https://python-poetry.org/docs/#installation))
- **pnpm**: For frontend package management ([installation guide](https://pnpm.io/installation))

### Quick Start

#### 1. Clone the Repository

```bash
git clone git@github.com:ncerny/basketball-video-analyzer.git
cd basketball-video-analyzer
```

#### 2. Set Up Backend

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

Backend will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

#### 3. Set Up Frontend

```bash
cd frontend
pnpm install
cp .env.example .env
pnpm dev
```

Frontend will be available at: http://localhost:5173

### Development Workflow

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines.

**Quick Reference:**

1. **Start work**: `git checkout -b feature/<issue-id>-description`
2. **Make changes**: Write code and tests
3. **Commit**: `git commit -m "type(scope): [issue-id] description"`
4. **Push**: `git push origin feature/<issue-id>-description`
5. **Create PR**: Title format `[<issue-id>] Description`
6. **Merge**: Squash and merge to `main`

## Project Structure

```
basketball-video-analyzer/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── schemas/        # Pydantic validation schemas
│   │   ├── api/            # API route handlers
│   │   ├── services/       # Business logic layer
│   │   └── ml/             # ML model inference
│   ├── tests/              # Backend tests (pytest)
│   └── pyproject.toml      # Poetry dependencies
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable React components
│   │   ├── pages/          # Page-level components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── store/          # Zustand state management
│   │   ├── api/            # Backend API client
│   │   └── types/          # TypeScript type definitions
│   ├── package.json        # pnpm dependencies
│   └── vite.config.ts      # Vite configuration
├── docs/                   # Project documentation
│   ├── architecture.md         # System architecture with diagrams
│   ├── implementation-plan.md  # Technical specs and roadmap
│   └── beads-structure.md      # Issue tracker organization
├── .beads/                 # Beads issue tracker database
├── CLAUDE.md               # Claude Code development guide
└── README.md               # This file
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)**: Development guide for Claude Code instances
- **[docs/architecture.md](./docs/architecture.md)**: System architecture with diagrams
- **[docs/implementation-plan.md](./docs/implementation-plan.md)**: Technical specs and roadmap
- **[docs/beads-structure.md](./docs/beads-structure.md)**: Issue tracker organization
- **[backend/README.md](./backend/README.md)**: Backend setup and API documentation
- **[frontend/README.md](./frontend/README.md)**: Frontend setup and development guide

## Development Phases

The project is organized into 6 phases:

### Phase 1: MVP Foundation ✅
- ✅ Project setup and infrastructure
- ✅ Video management and upload
- ✅ Manual annotation interface
- ✅ Player roster management
- ✅ Unified multi-video timeline

### Phase 2: Computer Vision (Current)
- Player detection with YOLO/RF-DETR
- Player tracking with ByteTrack/Norfair
- Jersey number recognition with SmolVLM2
- Batch-based processing with resume capability

### Phase 3: Play Recognition
- Automatic play detection
- Play classification
- User verification interface

### Phase 4: Search & Export
- Advanced search and filtering
- Clip generation
- Highlight reel creation

### Phase 5: Multi-Angle Features
- Enhanced multi-angle viewing
- Automatic video synchronization
- Advanced timeline features

### Phase 6: Desktop App
- Electron packaging
- Native desktop distribution
- Offline capabilities

## Issue Tracking

This project uses **Beads** for issue tracking instead of GitHub Issues. Issues are stored locally in the `.beads/` directory.

### View Issues

```bash
# Show ready-to-work issues (no blockers)
bd ready

# List all open issues
bd list --status open

# Show issue details
bd show <issue-id>

# Show project statistics
bd stats
```

### Work on Issues

```bash
# Create new issue
bd create "Task title" -t feature -p 0

# Update issue status
bd update <issue-id> --status in_progress

# Close completed issue
bd close <issue-id> --reason "Completed in PR #X"
```

Issue IDs follow the format: `bbva-<random>` (e.g., `bbva-a58df2`)

## Testing

### Backend Tests

```bash
cd backend
poetry run pytest                    # Run all tests
poetry run pytest --cov=app         # With coverage
poetry run pytest tests/test_main.py # Specific file
```

### Frontend Tests

```bash
cd frontend
pnpm test              # Run tests in watch mode
pnpm test:coverage     # With coverage
pnpm test:ui           # Open Vitest UI
```

## Contributing

### Commit Convention

All commits must follow Conventional Commits with Beads issue IDs:

**Format**: `<type>(<scope>): [<issue-id>] <description>`

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples**:
```bash
feat(backend): [bbva-a58df2] add video upload endpoint
fix(frontend): [bbva-b72f1a] fix timeline scrubber position
docs: [bbva-c93e4b] add API documentation for games endpoint
```

### Pull Requests

- Keep PRs small (< 400 lines when possible)
- Include clear description and screenshots for UI changes
- Reference the Beads issue ID in the title: `[bbva-xyz] Description`
- Ensure all tests pass before requesting review
- Use **Squash and Merge** when merging to `main`

## License

TBD

## Contact

TBD
