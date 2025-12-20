/**
 * Jersey Numbers API endpoints
 */

import { apiClient } from './client'

export interface AggregatedJerseyNumber {
  tracking_id: number
  jersey_number: number | null
  confidence: number
  total_readings: number
  valid_readings: number
  has_conflict: boolean
  all_numbers: number[]
}

export interface JerseyNumbersByTrack {
  video_id: number
  tracks: AggregatedJerseyNumber[]
  total_tracks: number
  tracks_with_numbers: number
}

export const jerseyNumbersAPI = {
  /**
   * Get aggregated jersey numbers by tracking ID for a video
   */
  async getByTrack(videoId: number): Promise<JerseyNumbersByTrack> {
    const response = await apiClient.get<JerseyNumbersByTrack>(
      `/videos/${videoId}/jersey-numbers/by-track`
    )
    return response.data
  },
}
