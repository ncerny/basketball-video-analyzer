/**
 * Detection API endpoints
 */

import type {
  DetectionJobRequest,
  DetectionJobResponse,
  DetectionStats,
  JobResponse,
  VideoDetectionsResponse,
} from '../types/api'
import { apiClient } from './client'

export const detectionAPI = {
  /**
   * Start player detection job for a video
   */
  async startDetection(
    videoId: number,
    request?: DetectionJobRequest
  ): Promise<DetectionJobResponse> {
    const response = await apiClient.post<DetectionJobResponse>(
      `/videos/${videoId}/detect`,
      request,
      {
        timeout: 120000, // 2 minutes - first call may need to download YOLO model
      }
    )
    return response.data
  },

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await apiClient.get<JobResponse>(`/jobs/${jobId}`)
    return response.data
  },

  /**
   * Cancel a running job
   */
  async cancelJob(jobId: string): Promise<void> {
    await apiClient.delete(`/jobs/${jobId}`)
  },

  /**
   * Get all detections for a video with optional filters
   */
  async getVideoDetections(
    videoId: number,
    params?: {
      frame_start?: number
      frame_end?: number
      min_confidence?: number
      limit?: number
    }
  ): Promise<VideoDetectionsResponse> {
    const response = await apiClient.get<VideoDetectionsResponse>(
      `/videos/${videoId}/detections`,
      { params }
    )
    return response.data
  },

  /**
   * Get detection statistics for a video
   */
  async getDetectionStats(videoId: number): Promise<DetectionStats> {
    const response = await apiClient.get<DetectionStats>(
      `/videos/${videoId}/detections/stats`
    )
    return response.data
  },

  /**
   * Delete all detections for a video
   */
  async deleteVideoDetections(videoId: number): Promise<void> {
    await apiClient.delete(`/videos/${videoId}/detections`)
  },
}
