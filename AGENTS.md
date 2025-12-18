# AI Agent Instructions

This repository uses AI assistants for development. All AI agents should read this file and the referenced documentation before making changes.

## Required Reading

1. **[CLAUDE.md](./CLAUDE.md)** - Project overview, architecture, database schema, technical challenges
2. **[docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)** - Development workflow, git conventions, coding standards

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

- **Multi-video timeline**: Games consist of multiple videos stitched into a continuous timeline
- **Annotations**: Reference `game_timestamp` (unified), not individual video timestamps
- **Local-first**: SQLite database, videos stored locally

Always verify your understanding by reading CLAUDE.md before making significant changes.
