/**
 * Annotations API endpoints
 */

import type { Annotation, CreateAnnotationDTO, UpdateAnnotationDTO } from '../types/api'
import { apiClient } from './client'

export const annotationsAPI = {
  /**
   * Get all annotations for a game
   */
  async getAnnotations(gameId: number): Promise<Annotation[]> {
    const response = await apiClient.get<Annotation[]>(`/games/${gameId}/annotations`)
    return response.data
  },

  /**
   * Get a single annotation by ID
   */
  async getAnnotation(id: number): Promise<Annotation> {
    const response = await apiClient.get<Annotation>(`/annotations/${id}`)
    return response.data
  },

  /**
   * Create a new annotation
   */
  async createAnnotation(data: CreateAnnotationDTO): Promise<Annotation> {
    const response = await apiClient.post<Annotation>('/annotations', data)
    return response.data
  },

  /**
   * Update an existing annotation
   */
  async updateAnnotation(id: number, data: UpdateAnnotationDTO): Promise<Annotation> {
    const response = await apiClient.patch<Annotation>(`/annotations/${id}`, data)
    return response.data
  },

  /**
   * Delete an annotation
   */
  async deleteAnnotation(id: number): Promise<void> {
    await apiClient.delete(`/annotations/${id}`)
  },
}
