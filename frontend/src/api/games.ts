/**
 * Games API endpoints
 */

import type { CreateGameDTO, Game, UpdateGameDTO } from '../types/api'
import { apiClient } from './client'

export const gamesAPI = {
  /**
   * Get all games
   */
  async getGames(): Promise<Game[]> {
    const response = await apiClient.get<Game[]>('/games')
    return response.data
  },

  /**
   * Get a single game by ID
   */
  async getGame(id: number): Promise<Game> {
    const response = await apiClient.get<Game>(`/games/${id}`)
    return response.data
  },

  /**
   * Create a new game
   */
  async createGame(data: CreateGameDTO): Promise<Game> {
    const response = await apiClient.post<Game>('/games', data)
    return response.data
  },

  /**
   * Update an existing game
   */
  async updateGame(id: number, data: UpdateGameDTO): Promise<Game> {
    const response = await apiClient.patch<Game>(`/games/${id}`, data)
    return response.data
  },

  /**
   * Delete a game
   */
  async deleteGame(id: number): Promise<void> {
    await apiClient.delete(`/games/${id}`)
  },
}
