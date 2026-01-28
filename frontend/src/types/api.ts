/**
 * API types and DTOs for backend communication
 */

// Game types
export interface Game {
  id: number
  name: string
  date: string // ISO date string
  location?: string
  home_team: string
  away_team: string
  created_at: string
  updated_at: string
}

export interface CreateGameDTO {
  name: string
  date: string
  location?: string
  home_team: string
  away_team: string
}

export interface UpdateGameDTO {
  name?: string
  date?: string
  location?: string
  home_team?: string
  away_team?: string
}

export interface GameList {
  games: Game[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// Video types
export interface Video {
  id: number
  game_id: number
  file_path: string
  r2_key?: string // Cloudflare R2 storage key
  thumbnail_path?: string
  duration_seconds: number
  fps: number
  resolution: string
  upload_date: string
  processed: boolean
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  recorded_at?: string
  sequence_order?: number
  game_time_offset?: number
}

// Stream URL response
export interface StreamUrlResponse {
  url: string
  expires_in: number // seconds
}

export interface VideoList {
  videos: Video[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface CreateVideoDTO {
  game_id: number
  file: File
}

// Player types
export interface Player {
  id: number
  name: string
  jersey_number: number
  team: string
  notes?: string
  created_at: string
}

export interface PlayerList {
  players: Player[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface CreatePlayerDTO {
  name: string
  jersey_number: number
  team: string
  notes?: string
}

export interface UpdatePlayerDTO {
  name?: string
  jersey_number?: number
  team?: string
  notes?: string
}

// Game Roster types
export interface GameRoster {
  id: number
  game_id: number
  player_id: number
  team_side: 'home' | 'away'
  jersey_number_override?: number
}

export interface CreateGameRosterDTO {
  game_id: number
  player_id: number
  team_side: 'home' | 'away'
  jersey_number_override?: number
}

export interface GameRosterList {
  rosters: GameRoster[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// Annotation types
export interface Annotation {
  id: number
  game_id: number
  title?: string
  description?: string
  game_timestamp_start: number
  game_timestamp_end: number
  annotation_type: 'play' | 'event' | 'note'
  confidence_score?: number
  verified: boolean
  created_by: 'ai' | 'user'
  created_at?: string
  updated_at?: string
}

export interface CreateAnnotationDTO {
  game_id: number
  title?: string
  description?: string
  game_timestamp_start: number
  game_timestamp_end: number
  annotation_type: 'play' | 'event' | 'note'
  confidence_score?: number
  verified?: boolean
  created_by: 'ai' | 'user'
}

export interface UpdateAnnotationDTO {
  title?: string
  description?: string
  game_timestamp_start?: number
  game_timestamp_end?: number
  annotation_type?: 'play' | 'event' | 'note'
  confidence_score?: number
  verified?: boolean
}

export interface AnnotationList {
  annotations: Annotation[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// Play types
export interface Play {
  id: number
  annotation_id: number
  play_type: 'basket' | 'miss' | 'turnover' | 'rebound' | 'foul' | 'substitution' | 'timeout'
  player_ids: number[]
  team: string
  points_scored?: number
  description?: string
}

export interface CreatePlayDTO {
  annotation_id: number
  play_type: 'basket' | 'miss' | 'turnover' | 'rebound' | 'foul' | 'substitution' | 'timeout'
  player_ids: number[]
  team: string
  points_scored?: number
  description?: string
}

// Error types
export class APIError extends Error {
  status?: number;
  code?: string;
  details?: unknown;

  constructor(
    message: string,
    status?: number,
    code?: string,
    details?: unknown
  ) {
    super(message)
    this.name = 'APIError'
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export interface ErrorResponse {
  message: string
  code?: string
  details?: unknown
}

// Detection types
export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Detection {
  id: number
  video_id: number
  frame_number: number
  player_id: number | null
  bbox: BoundingBox
  tracking_id: number
  confidence_score: number
}

export interface VideoDetectionsResponse {
  video_id: number
  total_detections: number
  detections: Detection[]
  frames_with_detections: number
}

export interface DetectionJobRequest {
  sample_interval?: number
  batch_size?: number
  confidence_threshold?: number
}

export interface DetectionJobResponse {
  job_id: string
  video_id: number
  message: string
}

export interface JobProgress {
  current: number
  total: number
  percentage: number
  message: string
}

export interface JobResponse {
  id: string
  job_type: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: JobProgress
  result: Record<string, unknown> | null
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  metadata: Record<string, unknown>
}

export interface DetectionStats {
  video_id: number
  total_frames_processed: number
  total_detections: number
  persons_detected: number
  balls_detected: number
  frames_with_detections: number
  avg_detections_per_frame: number
}
