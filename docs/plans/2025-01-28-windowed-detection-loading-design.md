# Windowed Detection Loading Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Load detections in windows around the current playback position instead of all at once, reducing initial load time and memory usage.

**Architecture:** Rolling window approach - fetch detections for current frame ± buffer, prefetch when approaching window boundary, replace on seeks.

**Tech Stack:** React hooks, existing detection API with frame_start/frame_end params

---

## Parameters

```typescript
const WINDOW_SIZE = 1000;        // frames (~33 seconds at 30fps)
const BUFFER_THRESHOLD = 300;    // frames from edge to trigger prefetch
```

At ~31 detections/frame, each window fetch is ~31,000 detections (~2.3 MB).

## State

```typescript
const [loadedRange, setLoadedRange] = useState<{start: number, end: number} | null>(null);
const [isLoadingDetections, setIsLoadingDetections] = useState(false);
const [detections, setDetections] = useState<Detection[]>([]);
```

## Fetch Logic

### 1. Video Load / Seek (Replace)

When video changes or user seeks outside current window:
- Calculate new window centered on target frame
- `newStart = max(0, targetFrame - WINDOW_SIZE/2)`
- `newEnd = targetFrame + WINDOW_SIZE/2`
- Fetch detections for `[newStart, newEnd]`
- Replace current detections with result
- Update `loadedRange`

### 2. Playback Approaching Edge (Extend & Trim)

On each frame update via `timeupdate` event:
- If `currentFrame > loadedRange.end - BUFFER_THRESHOLD`:
  - Prefetch next window `[loadedRange.end, loadedRange.end + WINDOW_SIZE]`
  - Merge with existing detections
  - Trim detections older than `currentFrame - WINDOW_SIZE`
  - Update `loadedRange`

- If `currentFrame < loadedRange.start + BUFFER_THRESHOLD`:
  - Prefetch previous window (for reverse playback)
  - Same merge/trim logic

### 3. Toggle Detections On

Same as video load - fetch window around current frame.

## Files to Modify

- `frontend/src/components/GameTimelinePlayer.tsx` - all changes here

## No Changes Needed

- Backend API (already supports `frame_start`, `frame_end`)
- Detection overlay rendering (receives detections as prop)
