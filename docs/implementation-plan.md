# Basketball Video Analyzer - Implementation Plan

## Project Overview
A local-first, multi-platform application for analyzing youth basketball game videos with semi-automatic player tracking and play tagging capabilities.

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐             │
│  │   Web    │  │  Desktop  │  │    Mobile    │             │
│  │ (React)  │  │ (Electron)│  │(React Native)│             │
│  └──────────┘  └───────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                          ↕ REST/WebSocket API
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  API Server (Python FastAPI or Node.js Express)    │     │
│  │  - Video management                                 │     │
│  │  - Annotation CRUD                                  │     │
│  │  - ML model orchestration                          │     │
│  │  - Real-time processing status                     │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                               ↕
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │   Video      │  │   Object    │  │   Player     │       │
│  │  Processing  │  │  Detection  │  │   Tracking   │       │
│  │  (OpenCV/    │  │  (YOLO-     │  │  (DeepSORT/  │       │
│  │   FFmpeg)    │  │   tiny/nano)│  │   ByteTrack) │       │
│  └──────────────┘  └─────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌─────────────┐                         │
│  │     Play     │  │   Jersey    │                         │
│  │Recognition   │  │   Number    │                         │
│  │  (Heuristics │  │ Recognition │                         │
│  │   + Rules)   │  │    (OCR)    │                         │
│  └──────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                               ↕
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │    Video     │  │  Metadata   │  │  Annotations │       │
│  │    Files     │  │  Database   │  │   Database   │       │
│  │ (Local FS)   │  │  (SQLite)   │  │   (SQLite)   │       │
│  └──────────────┘  └─────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI (async, fast, great for ML integration)
- **Video Processing**:
  - OpenCV for frame extraction and manipulation
  - FFmpeg for video encoding/decoding
- **ML/CV Libraries**:
  - YOLOv8-nano or YOLOv5-small (ultralytics) - lightweight, local-friendly
  - ByteTrack or DeepSORT for player tracking
  - EasyOCR or Tesseract for jersey number recognition
  - PyTorch (CPU mode for local inference)
- **Database**: SQLite with SQLAlchemy ORM
- **Task Queue**: Python asyncio or Celery with Redis (for background processing)

### Frontend
- **Web**: React 18+ with TypeScript
- **Desktop**: Electron with same React codebase
- **Mobile** (future): React Native or Capacitor
- **State Management**: Zustand or Redux Toolkit
- **Video Player**: Video.js or custom HTML5 with controls
- **UI Framework**: Tailwind CSS + shadcn/ui components

### Development Tools
- **Package Management**: Poetry (Python), pnpm (JavaScript)
- **API Documentation**: OpenAPI/Swagger (auto-generated from FastAPI)
- **Testing**: pytest (backend), Vitest + React Testing Library (frontend)
- **Build**: Vite (frontend), Docker (optional containerization)

## Database Schema

### Core Tables

**games**
- id (PRIMARY KEY)
- name
- date
- location
- home_team
- away_team
- created_at
- updated_at

**videos**
- id (PRIMARY KEY)
- game_id (FOREIGN KEY)
- file_path
- duration_seconds
- fps
- resolution
- upload_date
- processed (boolean)
- processing_status (enum: pending/processing/completed/failed)
- recorded_at (datetime from video metadata)
- sequence_order (int - order in game timeline, nullable until determined)
- game_time_offset (seconds - offset from game start, nullable until synced)

**players**
- id (PRIMARY KEY)
- name
- jersey_number
- team (team name/identifier)
- notes
- created_at

**game_rosters**
- id (PRIMARY KEY)
- game_id (FOREIGN KEY)
- player_id (FOREIGN KEY)
- team_side (enum: home/away)
- jersey_number_override (nullable - if player wore different number this game)

**annotations**
- id (PRIMARY KEY)
- game_id (FOREIGN KEY)
- game_timestamp_start (timestamp relative to game start)
- game_timestamp_end (timestamp relative to game start)
- annotation_type (enum: play/event/note)
- confidence_score (for ML-generated annotations)
- verified (boolean - user confirmed)
- created_by (enum: ai/user)

**annotation_videos**
- id (PRIMARY KEY)
- annotation_id (FOREIGN KEY)
- video_id (FOREIGN KEY)
- video_timestamp_start (timestamp in this specific video)
- video_timestamp_end (timestamp in this specific video)

**plays**
- id (PRIMARY KEY)
- annotation_id (FOREIGN KEY)
- play_type (enum: basket/miss/turnover/rebound/foul/substitution/timeout)
- player_ids (JSON array)
- team
- points_scored
- description

**player_detections**
- id (PRIMARY KEY)
- video_id (FOREIGN KEY)
- frame_number
- player_id (FOREIGN KEY, nullable)
- bbox_x, bbox_y, bbox_width, bbox_height
- tracking_id (for cross-frame tracking)
- confidence_score

## Core Features & Implementation

### Phase 1: Foundation (MVP)
**Goal**: Basic video upload, storage, and manual annotation

#### 1.1 Video Management System
- Upload video files (MP4, MOV, AVI)
- Store locally with organized directory structure: `videos/{game_id}/{video_id}/`
- Extract metadata (duration, fps, resolution) using FFmpeg
- Generate thumbnails for video preview
- Create game entries with basic metadata

#### 1.2 Unified Game Video Player
**Core Concept**: User watches a "game" not individual videos. Multiple videos are seamlessly stitched into a continuous timeline.

**Video Sequencing & Synchronization**:
- Extract `recorded_at` timestamp from video file metadata (creation time, modification time)
- **Manual sequencing UI**: Allow user to drag/drop videos into correct chronological order
- **Automatic sequencing** (Phase 2): OCR scoreboard/game clock to determine sequence and sync points
  - Extract game clock from frames (e.g., "8:45 Q2")
  - Match overlapping time segments across videos
  - Calculate `game_time_offset` for each video
- Handle gaps between videos (show placeholder "No video coverage for this period")

**Unified Playback Experience**:
- Single continuous timeline representing the entire game
- Automatically transition between videos as playback crosses video boundaries
- Visual indicators showing which video is currently playing
- Smooth handoff between videos (pre-buffer next video)
- Game timestamp display (e.g., "Q2 8:45") instead of raw video time
- Allow manual override of sequence/timing via UI

**Handling Overlapping Videos** (same timeframe, different angles):
- **MVP approach**: Default to primary video (first uploaded, or user-selected)
- **Future enhancement**: Allow angle switching during overlapping periods
- UI shows "Multiple angles available" indicator during overlap
- Quick-switch button or dropdown to change active video
- Timeline shows overlapping segments visually

**Player Controls**:
- Play/pause, frame-by-frame navigation (←/→ keys)
- Playback speed control (0.25x, 0.5x, 1x, 2x)
- Jump between videos (if user wants to see specific angle)
- Timeline scrubber with video segment indicators (color-coded by video source)
- Video source selector for overlapping periods

#### 1.3 Manual Annotation Tools
- Click timeline to add annotation markers (at game time, not video time)
- Form to specify:
  - Play type (dropdown)
  - Players involved (multi-select from game roster)
  - Start/end time (with fine-tuning controls)
  - Notes
- **Multi-video handling**:
  - System automatically determines which video(s) contain the annotated timeframe
  - Creates entries in `annotation_videos` for all relevant videos
  - If timeframe spans gap between videos, warn user
  - If overlapping videos exist, annotation references all angles
- Edit/delete annotations
- Color-coded markers on timeline by play type
- Show video source(s) for each annotation in UI

#### 1.4 Player Management
- Add players to a game (name, number, team)
- Edit player information
- Visual roster display

### Phase 2: Computer Vision Integration
**Goal**: Semi-automatic player detection and tracking

#### 2.1 Player Detection Pipeline
```python
# Workflow:
1. Extract frames from video (every 2-3 frames for efficiency)
2. Run YOLOv8-nano to detect people/players
3. Filter detections (court boundaries, size filters)
4. Track players across frames using ByteTrack
5. Assign tracking IDs
6. Store detections in database
7. Present to user for confirmation
```

**Implementation approach**:
- Background processing job triggered after video upload
- Progress indicator in UI
- Generate bounding boxes for each detected player
- Store tracking data with confidence scores
- UI overlay showing detections on video player

#### 2.2 Player Identification
- Jersey number OCR on detected player bounding boxes
- User interface to:
  - Review detected jersey numbers
  - Assign tracking IDs to player roster entries
  - Manually correct misidentifications
- Color-coding or labels on video overlay showing player names

#### 2.3 Court Detection & Homography
- Detect basketball court boundaries
- Identify key points (hoops, free-throw line, three-point line)
- Optional: Transform perspective for bird's-eye view
- Use court position for filtering false detections

#### 2.4 Batch-Based Processing Pipeline

**Problem**: Long-running video processing is fragile - if interrupted, all progress is lost. OCR is slow (~15s/frame) and blocks detection progress.

**Solution**: Process videos in batches with database checkpoints, enabling resume after interruption.

##### Architecture: Batch-Based Task DAG

```
Video Processing Job
│
├─ Batch 1: [Detect frames 0-29] ──→ [OCR batch 1 detections]
├─ Batch 2: [Detect frames 30-59] ──→ [OCR batch 2 detections]  
├─ Batch 3: [Detect frames 60-89] ──→ [OCR batch 3 detections]
└─ ...

Each arrow = dependency (OCR blocked only by its detection batch)
Batches are independent of each other
```

##### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Sequential** | Detection → OCR → next batch | Local, single GPU, predictable |
| **Pipeline** | Detection N+1 while OCR N | Local, if spare cycles |
| **Distributed** | Separate nodes for detection/OCR | Future, multi-machine |

##### Database: Processing Batches Table

```sql
CREATE TABLE processing_batches (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    batch_index INTEGER,              -- 0, 1, 2, ...
    frame_start INTEGER,
    frame_end INTEGER,
    detection_status TEXT,            -- pending, processing, completed, failed
    ocr_status TEXT,                  -- pending, processing, completed, failed, skipped
    detection_completed_at TIMESTAMP,
    ocr_completed_at TIMESTAMP,
    created_at TIMESTAMP
);
```

##### Code Structure

```
backend/app/services/
├── batch_processor.py         # Individual batch operations
│   ├── DetectionBatchProcessor   # Run detection on frame range
│   └── OCRBatchProcessor         # Run OCR on detection batch
├── batch_orchestrator.py      # Coordinates batch execution
│   ├── SequentialOrchestrator    # Mode 1: local sequential
│   ├── PipelineOrchestrator      # Mode 2: local pipeline  
│   └── DistributedOrchestrator   # Mode 3: distributed
└── detection_pipeline.py      # Refactored to use orchestrator
```

##### Resume Logic

1. On job start, query `processing_batches` for video
2. Find first batch where `detection_status != 'completed'`
3. Resume from that batch
4. Skip already-completed OCR batches

##### Configuration

```python
# config.py
batch_size_frames: int = 30
execution_mode: Literal["sequential", "pipeline", "distributed"] = "sequential"
```

##### Benefits

1. **Resilient**: Interrupted jobs resume from last checkpoint
2. **Flexible**: Swap execution mode without changing processors
3. **Observable**: Batch table provides granular progress visibility
4. **Scalable**: Path to distributed processing on multiple nodes

### Phase 3: Play Recognition
**Goal**: Automatic play detection and categorization

#### 3.1 Event Detection (Rule-Based)
- **Basket detection**:
  - Track ball trajectory toward hoop
  - Detect player near hoop
  - Infer from score changes (if available)
- **Turnover detection**:
  - Ball possession changes (tracking ID changes)
  - Ball goes out of bounds (court boundary detection)
- **Rebound detection**:
  - Ball trajectory after miss
  - Player proximity to hoop
- **Substitution detection**:
  - Player enters/exits tracked region
  - Duration on court changes

#### 3.2 Play Suggestion System
- ML model suggests play type and timestamp
- Confidence score for each suggestion
- User reviews and confirms/corrects
- Feedback loop to improve suggestions over time

#### 3.3 Play Tagging Interface
- Automatic suggestions appear as "unverified" tags
- User can:
  - Approve (mark as verified)
  - Edit details (players, time, type)
  - Reject (delete)
- Bulk approval for high-confidence detections

### Phase 4: Search & Playback
**Goal**: Efficiently find and watch specific plays

#### 4.1 Search & Filter System
- Filter by:
  - Player name/number
  - Play type
  - Team
  - Time range
  - Verified vs unverified
- Full-text search in notes/descriptions
- Save common filter combinations

#### 4.2 Play Clip Generation
- Generate video clips for individual plays
- Start 5-10 seconds before, end 5-10 seconds after
- Export clips as separate files (optional)
- Create playlist of filtered plays

#### 4.3 Highlight Reel Generator
- Select multiple plays
- Concatenate into single video
- Add transitions (optional)
- Export as new video file

### Phase 5: Advanced Multi-Angle Features
**Goal**: Enhanced multi-angle viewing experience

**Note**: Basic multi-video sequencing and manual synchronization is part of MVP (Phase 1). This phase focuses on advanced features.

#### 5.1 Automatic Video Alignment
- **Audio cross-correlation**: Sync videos by matching audio waveforms
- **Visual event detection**: Detect common events across videos (jump ball, baskets, whistles)
- **Scoreboard OCR sync**: Match game clock across different video sources
- **Confidence scoring**: Show sync quality and allow manual adjustment

#### 5.2 Advanced Multi-Angle Playback
- **Seamless angle switching**: Hotkeys to switch between angles during playback
- **Picture-in-picture mode**: Show multiple angles simultaneously
- **Split-screen view**: Side-by-side comparison of angles
- **Synchronized scrubbing**: When moving timeline in one video, all angles follow
- **Angle preference memory**: Remember which angle user prefers for different plays

## Project Structure

```
basketball-video-analyzer/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── config.py               # Configuration
│   │   ├── database.py             # DB connection
│   │   ├── models/                 # SQLAlchemy models
│   │   │   ├── game.py
│   │   │   ├── video.py
│   │   │   ├── player.py
│   │   │   ├── game_roster.py
│   │   │   ├── annotation.py
│   │   │   ├── annotation_video.py
│   │   │   └── detection.py
│   │   ├── schemas/                # Pydantic schemas
│   │   │   └── ...
│   │   ├── api/                    # API routes
│   │   │   ├── games.py
│   │   │   ├── videos.py
│   │   │   ├── players.py
│   │   │   ├── annotations.py
│   │   │   └── processing.py
│   │   ├── services/               # Business logic
│   │   │   ├── video_processor.py
│   │   │   ├── player_detector.py
│   │   │   ├── tracker.py
│   │   │   ├── play_recognizer.py
│   │   │   └── clip_generator.py
│   │   └── ml/                     # ML models
│   │       ├── yolo_detector.py
│   │       ├── tracker.py
│   │       ├── jersey_ocr.py
│   │       └── models/             # Pre-trained model files
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoPlayer/
│   │   │   ├── Timeline/
│   │   │   ├── AnnotationForm/
│   │   │   ├── PlayerRoster/
│   │   │   ├── PlayList/
│   │   │   └── SearchFilters/
│   │   ├── pages/
│   │   │   ├── GameList.tsx
│   │   │   ├── GameDetail.tsx
│   │   │   ├── VideoAnalysis.tsx
│   │   │   └── Settings.tsx
│   │   ├── hooks/
│   │   ├── store/                  # State management
│   │   ├── api/                    # API client
│   │   ├── types/
│   │   └── utils/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── desktop/                        # Electron wrapper
│   ├── main.js
│   ├── preload.js
│   └── package.json
├── mobile/                         # Future: React Native
├── docs/
│   ├── API.md
│   ├── SETUP.md
│   └── USER_GUIDE.md
├── docker-compose.yml              # Optional
└── README.md
```

## Development Roadmap

### Milestone 1: MVP (4-6 weeks)
- [ ] Backend API server with basic endpoints
- [ ] SQLite database with core schema
- [ ] Video upload and storage
- [ ] Web frontend with video player
- [ ] Manual annotation tools
- [ ] Player management

### Milestone 2: Computer Vision (4-6 weeks)
- [ ] YOLOv8 integration for player detection
- [ ] ByteTrack implementation for tracking
- [ ] Background processing pipeline
- [ ] Detection review UI
- [ ] Jersey number OCR (basic)

### Milestone 3: Play Recognition (3-4 weeks)
- [ ] Rule-based event detection
- [ ] Play suggestion system
- [ ] Verification UI for suggested plays
- [ ] Basic statistics dashboard

### Milestone 4: Search & Export (2-3 weeks)
- [ ] Advanced search and filtering
- [ ] Play clip generation
- [ ] Highlight reel creator
- [ ] Export functionality

### Milestone 5: Desktop App (2 weeks)
- [ ] Electron packaging
- [ ] Native file system integration
- [ ] Installer creation

### Milestone 6: Polish & Optimization (2-3 weeks)
- [ ] Performance optimization
- [ ] ML model fine-tuning
- [ ] User testing and feedback
- [ ] Documentation

## Key Technical Challenges & Solutions

### Challenge 0: Unified Game Timeline with Multiple Videos
**Problem**: Seamlessly stitch multiple videos into a continuous game timeline, handle gaps, and manage overlapping coverage.

**Solution**:
- **Video Metadata Extraction**:
  - Use FFmpeg/OpenCV to extract creation timestamp from video files
  - Store as `recorded_at` in database
  - Extract actual video duration and frame rate

- **Timeline Construction Algorithm**:
  ```
  1. Sort videos by `recorded_at` timestamp
  2. Calculate initial `sequence_order` based on chronological order
  3. For each video, calculate `game_time_offset`:
     - First video: offset = 0
     - Subsequent videos: offset = previous_offset + previous_duration (if no gap)
     - If gap detected: offset = previous_offset + previous_duration + gap_duration
  4. Detect overlaps: if next_video.recorded_at < (prev_recorded_at + prev_duration)
  ```

- **Playback State Management**:
  - Frontend maintains: `currentGameTime` (seconds from game start)
  - Convert `currentGameTime` to specific video + video timestamp:
    ```javascript
    function getActiveVideo(gameTime, videos) {
      return videos.find(v =>
        gameTime >= v.game_time_offset &&
        gameTime < v.game_time_offset + v.duration
      )
    }
    ```
  - Pre-load next video when approaching boundary (5 seconds before)
  - Handle video transitions by pausing briefly or cross-fading

- **Annotation Mapping**:
  - Annotations stored with `game_timestamp_start/end` (relative to game)
  - When displaying: map to `video_timestamp` via `annotation_videos` table
  - Allow same annotation to reference multiple videos (different angles)

- **UI Timeline Visualization**:
  - Single timeline bar representing full game
  - Color-coded segments showing which video covers each period
  - Gray segments for gaps in coverage
  - Striped/patterned segments for overlapping coverage
  - Hover to show video source details

### Challenge 1: Local ML Performance
**Problem**: Running computer vision models locally on consumer hardware
**Solution**:
- Use lightweight models (YOLOv8-nano, ~6MB)
- Process videos at lower FPS (sample every 2-3 frames)
- Implement frame batching for GPU efficiency (if available)
- Allow background processing with progress tracking
- Cache processed results

### Challenge 2: Player Re-identification
**Problem**: Tracking same player across different videos/games
**Solution**:
- Use jersey numbers as primary identifier
- Store player appearance features (color histograms, pose)
- Manual confirmation system for ambiguous cases
- Build player database over time

### Challenge 3: Occlusion & Crowding
**Problem**: Players blocking each other, difficult to track
**Solution**:
- Use robust tracking algorithm (ByteTrack handles occlusion well)
- Implement track recovery after occlusion
- Allow manual correction of tracking IDs
- Use temporal smoothing for predictions

### Challenge 4: Varied Video Quality
**Problem**: Parent-recorded videos have varying quality, angles, lighting
**Solution**:
- Adaptive detection thresholds based on video quality
- Pre-processing: stabilization, brightness adjustment
- User feedback to tune parameters per video
- Graceful degradation (lower confidence, more manual input)

### Challenge 5: Play Boundary Detection
**Problem**: Determining exact start/end of a play
**Solution**:
- Use generous time buffers (±10 seconds)
- Allow user to trim in UI
- Learn from user corrections
- Visual indicators (ball movement, whistle detection)

## Optional Advanced Features (Future)

### AI/ML Enhancements
- Fine-tune models on youth basketball footage
- Player pose estimation for skill analysis
- Shot arc analysis (make/miss prediction)
- Defensive formation recognition

### Analytics Dashboard
- Player statistics (points, rebounds, assists)
- Heat maps (where players spend time)
- Shot charts
- Team performance metrics over time

### Collaboration Features
- Share annotations with other users
- Cloud backup (optional, with encryption)
- Export to common formats (XML, JSON, CSV)

### Mobile App Features
- Live game recording with auto-upload
- Quick play tagging during game
- Offline viewing of processed games

## Success Metrics

### MVP Success Criteria
1. Successfully upload and play back videos
2. Manually annotate 50+ plays in a full game within 30 minutes
3. Create player roster and filter plays by player
4. Generate clip of specific play

### CV Integration Success Criteria
1. Detect 80%+ of players on court with >70% confidence
2. Track individual players across 90%+ of frames
3. Correctly identify jersey numbers 60%+ of time
4. Reduce manual annotation time by 50%

### Overall Success
- Parents/coaches can analyze a full game in under 1 hour
- Find and watch all plays by a specific player in <2 minutes
- Generate highlight reel of a player's best plays in <5 minutes

## Next Steps to Begin Implementation

### Step 1: Project Initialization
- Initialize Python backend with FastAPI
- Initialize React frontend with Vite + TypeScript
- Set up SQLite database with Alembic migrations
- Create git repository with proper .gitignore
- Set up development environment (Docker optional)

### Step 2: Database Schema Implementation
- Create SQLAlchemy models for all tables:
  - `games`, `videos`, `players`, `game_rosters`
  - `annotations`, `annotation_videos`, `plays`
  - `player_detections`
- Write Alembic migration scripts
- Create seed data for testing

### Step 3: Backend Core Services
- **Video processing service**:
  - Video upload handler (multipart form data)
  - FFmpeg metadata extraction (duration, fps, resolution, recorded_at)
  - Video storage organization (filesystem structure)
  - Thumbnail generation
- **Game timeline service**:
  - Video sequencing algorithm
  - Game time offset calculation
  - Overlap detection
  - Gap detection and handling
- **API endpoints**:
  - Games CRUD
  - Videos CRUD (with timeline data)
  - Players CRUD
  - Game rosters management
  - Annotations CRUD (with multi-video support)

### Step 4: Frontend Core Components
- **Game Timeline Player** (most complex component):
  - Unified timeline state management
  - Multi-video playback engine
  - Video transition handling
  - Game time ↔ video time conversion
  - Timeline visualization with video segments
  - Playback controls with frame-by-frame navigation
- **Video Sequencing UI**:
  - Drag-and-drop video ordering
  - Manual time offset adjustment
  - Video gap visualization
- **Annotation Interface**:
  - Timeline marker creation/editing
  - Annotation form with player selection
  - Multi-video annotation display
- **Player/Roster Management**:
  - Player database UI
  - Game roster assignment
  - Jersey number management

### Step 5: MVP Integration & Testing
- Connect frontend to backend APIs
- End-to-end testing with real basketball footage
- Handle edge cases:
  - Single video games
  - Multi-video with gaps
  - Overlapping videos
  - Very short/long videos
- Performance optimization for large video files
- User documentation for MVP features

### Step 6: Computer Vision Foundation (Post-MVP)
- Download and test YOLOv8-nano model locally
- Set up OpenCV and video frame extraction
- Create background processing job system
- Implement basic player detection pipeline
- Test on sample basketball footage

### Step 7: Iterate Based on Feedback
- User testing with parents/coaches
- Gather feedback on:
  - Video sequencing UX
  - Annotation workflow
  - Performance with typical hardware
- Prioritize next features
- Optimize pain points
