/**
 * Videos API endpoints
 */

import type { Video } from '../types/api'
import { apiClient } from './client'

export const videosAPI = {
  /**
   * Get all videos for a game
   */
  async getVideos(gameId: number): Promise<Video[]> {
    const response = await apiClient.get<Video[]>(`/games/${gameId}/videos`)
    return response.data
  },

  /**
   * Get a single video by ID
   */
  async getVideo(id: number): Promise<Video> {
    const response = await apiClient.get<Video>(`/videos/${id}`)
    return response.data
  },

  /**
   * Upload a video for a game
   */
  async uploadVideo(gameId: number, file: File, onProgress?: (progress: number) => void): Promise<Video> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('game_id', gameId.toString())

    const response = await apiClient.post<Video>('/video-upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(percentCompleted)
        }
      },
    })

    return response.data
  },

  /**
   * Delete a video
   */
  async deleteVideo(id: number): Promise<void> {
    await apiClient.delete(`/videos/${id}`)
  },
}
