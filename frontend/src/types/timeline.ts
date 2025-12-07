/**
 * Timeline types for unified game video playback
 */

/**
 * Video with timeline synchronization data
 */
export interface Video {
  id: number
  game_id: number
  file_path: string
  duration_seconds: number
  fps: number
  resolution: string
  recorded_at?: string
  sequence_order?: number
  game_time_offset?: number // Offset from game start in seconds
}

/**
 * Timeline annotation
 */
export interface Annotation {
  id: number
  game_id: number
  game_timestamp_start: number // Seconds from game start
  game_timestamp_end: number // Seconds from game start
  annotation_type: 'play' | 'event' | 'note'
  confidence_score?: number
  verified: boolean
  created_by: 'ai' | 'user'
}

/**
 * Result of converting game time to video-specific time
 */
export interface VideoTimeResult {
  video: Video | null
  videoTime: number
  isInGap: boolean
}

/**
 * Video segment in the timeline
 */
export interface VideoSegment {
  video: Video
  startTime: number // Game time when this video starts
  endTime: number // Game time when this video ends
}

/**
 * Timeline state
 */
export interface TimelineState {
  currentGameTime: number
  currentVideoId: number | null
  videos: Video[]
  isPlaying: boolean
  playbackRate: number
  annotations: Annotation[]
}

/**
 * Timeline actions
 */
export interface TimelineActions {
  setGameTime: (time: number) => void
  play: () => void
  pause: () => void
  setPlaybackRate: (rate: number) => void
  loadVideos: (videos: Video[]) => void
  loadAnnotations: (annotations: Annotation[]) => void
  seekToGameTime: (time: number) => void
  getCurrentVideoTime: () => VideoTimeResult
  getGameTimeFromVideo: (videoId: number, videoTime: number) => number
  reset: () => void
}

/**
 * Complete timeline store type
 */
export type TimelineStore = TimelineState & TimelineActions
