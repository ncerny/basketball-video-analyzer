# AI Agent Instructions - Optimized

## Required Reading
1. **CLAUDE_OPTIMIZED.md** - Project overview (use this version for smaller context)
2. **docs/CONTRIBUTING.md** - Development workflow when needed

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Project** | Basketball Video Analyzer - youth basketball game analysis |
| **Backend** | Python 3.11+, FastAPI, SQLAlchemy, SQLite |
| **Frontend** | React 18+, TypeScript, Mantine UI, Zustand |
| **ML** | YOLOv8-nano for player detection |
| **Issue Tracker** | Beads (`bd` CLI) - NOT GitHub Issues |
| **Git** | Trunk-based development - commit directly to `main` |
| **Commits** | `type(scope): [bbva-xxx] description` |

## Workflow
1. Check for work: `bd ready`
2. Claim issue: `bd update <id> --status=in_progress`
3. Make changes and test
4. Commit with issue ID: `git commit -m "type(scope): [bbva-xxx] description"`
5. Push: `git push origin main`
6. Close issue: `bd close <id> --reason="description"`
7. Sync beads: `bd sync`

## Architecture Notes
- **Multi-video timeline**: Games consist of multiple videos stitched into unified timeline
- **Annotations**: Reference `game_timestamp` (unified), not individual video timestamps
- **Local-first**: SQLite database, videos stored locally

## Context Optimization
This repository uses `.opencodeignore` to exclude large files and prevent quota issues:
- Video files and directories
- Dependencies (node_modules, venv)
- Build artifacts and cache
- Database files
- Lock files

Always verify understanding with CLAUDE_OPTIMIZED.md before making changes.