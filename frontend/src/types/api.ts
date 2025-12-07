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

// Video types
export interface Video {
  id: number
  game_id: number
  file_path: string
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

// Annotation types
export interface Annotation {
  id: number
  game_id: number
  game_timestamp_start: number
  game_timestamp_end: number
  annotation_type: 'play' | 'event' | 'note'
  confidence_score?: number
  verified: boolean
  created_by: 'ai' | 'user'
}

export interface CreateAnnotationDTO {
  game_id: number
  game_timestamp_start: number
  game_timestamp_end: number
  annotation_type: 'play' | 'event' | 'note'
  confidence_score?: number
  verified?: boolean
  created_by: 'ai' | 'user'
}

export interface UpdateAnnotationDTO {
  game_timestamp_start?: number
  game_timestamp_end?: number
  annotation_type?: 'play' | 'event' | 'note'
  confidence_score?: number
  verified?: boolean
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
  constructor(
    message: string,
    public status?: number,
    public code?: string,
    public details?: unknown
  ) {
    super(message)
    this.name = 'APIError'
  }
}

export interface ErrorResponse {
  message: string
  code?: string
  details?: unknown
}
