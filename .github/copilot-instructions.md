# Copilot Instructions

This is the Basketball Video Analyzer project - a local-first application for analyzing youth basketball game videos.

## Required Reading

- **[/CLAUDE.md](/CLAUDE.md)** - Project overview, architecture, database schema
- **[/docs/CONTRIBUTING.md](/docs/CONTRIBUTING.md)** - Development workflow, git conventions, coding standards

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Backend** | Python 3.11+, FastAPI, SQLAlchemy, SQLite |
| **Frontend** | React 18+, TypeScript, Mantine UI, Zustand |
| **Testing** | pytest (backend), Vitest (frontend) |
| **Issue Tracker** | Beads (not GitHub Issues) - use `bd` CLI |
| **Git Workflow** | Trunk-based: commit directly to `main` |
| **Commits** | `type(scope): [bbva-xxx] description` |

## Key Patterns

- Multi-video timeline stitching for games
- YOLOv8-nano for player detection
- Annotations reference game time, not video time
- SQLite for local-first data storage

Always read the full context in CLAUDE.md before making significant changes.
