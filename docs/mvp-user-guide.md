# MVP User Guide

A practical handbook for running and using the Basketball Video Analyzer Minimum Viable Product (Phase 1). This document targets coaches, parents, and internal testers who need to stand up the stack locally and work through the core workflows (games, videos, rosters, and annotations).

---

## 1. What You Get in the MVP

| Capability | Where to Use It | Notes |
| --- | --- | --- |
| Game management | **Games** page | Create, view, and delete games; jump to details or analysis.
| Player database | **Players** page | Maintain a global roster, filter by team, edit/delete entries.
| Game rosters | **Game Detail** page | Assign players to the home/away roster for a game.
| Video upload & sequencing | **Game Detail** page | Upload MP4/MOV files, inspect metadata, drag-drop order, fix gaps/overlaps.
| Unified playback | **Video Analysis** page | Seamless multi-video playback with timeline, frame stepping, keyboard shortcuts.
| Manual annotations | **Video Analysis → Annotation Panel** | Create/edit/delete/verify annotations tied to game time.

> Everything runs locally. Future phases (computer vision, auto play detection, search/export, etc.) build on these foundations but are **not** covered here.

---

## 2. System Requirements

| Layer | Requirement |
| --- | --- |
| Backend | macOS/Linux, Python 3.11+, Poetry, FFmpeg (CLI), SQLite (bundled). |
| Frontend | Node.js 18+ (or 20+), pnpm 9+, modern browser (Edge/Chrome/Firefox/Safari). |
| Video files | H.264 MP4/MOV recommended; ensure enough disk space under `backend/videos`. |

Optional but helpful: `pyenv`, `direnv`, and a GPU is **not** required for Phase 1.

---

## 3. Environment Setup

### 3.1 Backend (FastAPI)

1. **Install dependencies**
   ```bash
   cd backend
   poetry install --without ml  # add --with ml when you need YOLO/EasyOCR
   ```
2. **Configure environment (optional)**
   - Copy `.env.example` to `.env` if you want to override defaults (database URL, `VIDEO_STORAGE_PATH`, etc.).
   - Defaults come from [`app/config.py`](../backend/app/config.py).
3. **Run migrations**
   ```bash
   poetry run alembic upgrade head
   ```
4. **Start the API**
   ```bash
   poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. **Verify**
   - Swagger UI: http://localhost:8000/docs
   - Redoc: http://localhost:8000/redoc
6. **Video storage**
   - Files land under `backend/videos/game_<id>/...` by default. Make sure the folder is writable and has space.

> **Port busy?** Run `lsof -ti:8000 | xargs kill -9` to free it before restarting Uvicorn.

### 3.2 Frontend (React + Vite)

1. **Install dependencies**
   ```bash
   cd frontend
   pnpm install
   ```
2. **Configure API base URL**
   ```bash
   cp .env.example .env
   # edit VITE_API_URL if your backend is not on http://localhost:8000
   ```
3. **Start dev server**
   ```bash
   pnpm dev
   ```
   - App serves at http://localhost:5173 (or the port Vite selects).
4. **Build for production** (optional)
   ```bash
   pnpm build && pnpm preview
   ```

### 3.3 Seed Sample Data (Optional, for demo/testing)

Run the async seed script to pre-populate games, players, rosters, videos, and annotations:

```bash
cd backend
poetry run python scripts/seed_data.py
```

- The script wipes existing records, so only run it on non-production data.
- Sample videos reference placeholder paths (`/data/videos/...`). Upload real footage to test playback.

---

## 4. Core Workflows

### 4.1 Manage Games

1. Navigate to **Games** (root route).
2. Click **New Game** to open the modal.
3. Provide name, date, home/away teams, and optional location; submit.
4. Cards show key meta plus action buttons:
   - **Details** → Game detail page
   - **Analyze** → Opens the Video Analysis workspace
   - **Trash icon** → Delete (with confirmation)

Games are sorted newest-first. Errors (API/network) appear in a dismissible alert.

### 4.2 Maintain the Player Database

1. Go to **Players**.
2. Use the search box or team filter to find entries fast.
3. Click **Add Player** to capture name, jersey number, team, and notes.
4. Use the table actions to edit (prefills modal) or delete individuals.

Players stay global—assign them to specific games later via the roster panel.

### 4.3 Build Game Rosters

1. Open a game&#8217;s **Details** page.
2. The **Game Roster** section shows separate cards for home/away teams.
3. Click **Add Player**, choose the team side, then select any player not already on the roster.
4. Remove players via the **Remove** button beside each entry.

Rosters are required for tagging plays with the right athletes and teams down the line.

### 4.4 Upload & Sequence Videos

1. From the **Game Detail → Videos** section, click **Upload Video** (accepts `video/*`).
2. Once processed, each card displays filename, duration, resolution, and processing status.
3. Use the trash icon to remove an unwanted clip.
4. Launch the **Video Sequencer** when multiple videos exist:
   - Drag rows or use Up/Down buttons to reorder.
   - Edit start offsets (seconds from game start) to insert gaps or overlaps deliberately.
   - **Sequence from timestamps** calls the backend timeline sequencer to derive order from `recorded_at` metadata.
   - **Remove gaps** aligns videos back-to-back.
   - **Save order** persists `sequence_order` + `game_time_offset`.
   - Coverage bar highlights gaps (grey) and overlaps (striped) so you can visually audit timeline health.

### 4.5 Analyze a Game (Unified Playback)

1. Click **Analyze** on a game card (or `/games/:id/analysis`).
2. The header shows matchup info and the active video&#8217;s resolution/FPS/duration.
3. The **GameTimelinePlayer** provides:
   - Multi-angle playback with seamless switching.
   - Keyboard shortcuts (`Space` play/pause, `←/→` frame step, `J/L` skip 10s, `↑/↓` speed).
   - Adjustable playback speeds (0.25×–2×).
   - Current time vs. total coverage.
   - Timeline bar with segment coloring and annotation markers.
4. Toggle the annotation side panel with **Show/Hide Annotations** to maximize video area.

### 4.6 Capture Manual Annotations

1. In Video Analysis, keep the annotation panel visible.
2. **Annotation List** features filters (type, verified-only), sorting, duration labels, and quick `Verify / Edit / Delete` actions. Clicking any entry seeks the timeline to its start.
3. Click **New Annotation** to open the form:
   - Optionally add title + description.
   - Choose type (`play`, `event`, `note`).
   - Capture start/end times from the current timeline position or type them in `MM:SS.ms`.
   - Provide a confidence score (0–1) if relevant and mark as verified when reviewing.
   - The footer shows how many videos cover the selected time span.
4. Submit to save; edit mode reuses the same form.

Annotations immediately update both the panel and the timeline markers, keeping playback and metadata in sync.

---

## 5. Troubleshooting & Tips

| Issue | Quick Fix |
| --- | --- |
| **`poetry run uvicorn` fails: port in use** | Free port 8000 (`lsof -ti:8000 | xargs kill -9`) and restart. |
| **Uploads fail for larger files** | Confirm backend `video_storage_path` has sufficient space and macOS privacy permissions allow disk writes. |
| **FFmpeg errors during metadata extraction** | Install FFmpeg via `brew install ffmpeg` (or ensure it is on `$PATH`). Check console output for stderr details. |
| **Frontend cannot reach API (`CORS` or 404)** | Verify `VITE_API_URL` matches the backend base URL and the API is running. |
| **Annotations not visible on timeline** | Ensure at least one video is sequenced (has `sequence_order` and `game_time_offset`) and refresh the analysis page to reload timeline data. |
| **Sample seed videos missing** | The seed script uses placeholder paths; upload actual clips per game to test playback. |

---

## 6. Next Steps

- Keep this guide close when onboarding new testers or demonstrating the MVP.
- When Phase 2 (computer vision) begins, we will extend this document with detection review workflows and troubleshooting notes for YOLO/ByteTrack.

Need help or spot an omission? Add an issue in Beads (e.g., `bd create "Docs: clarify annotation workflow" -t chore -p 2`) and link back to this guide.
