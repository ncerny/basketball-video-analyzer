/**
 * Players API endpoints
 */

import type { CreatePlayerDTO, Player, UpdatePlayerDTO } from '../types/api'
import { apiClient } from './client'

export const playersAPI = {
  /**
   * Get all players
   */
  async getPlayers(): Promise<Player[]> {
    const response = await apiClient.get<Player[]>('/players')
    return response.data
  },

  /**
   * Get a single player by ID
   */
  async getPlayer(id: number): Promise<Player> {
    const response = await apiClient.get<Player>(`/players/${id}`)
    return response.data
  },

  /**
   * Create a new player
   */
  async createPlayer(data: CreatePlayerDTO): Promise<Player> {
    const response = await apiClient.post<Player>('/players', data)
    return response.data
  },

  /**
   * Update an existing player
   */
  async updatePlayer(id: number, data: UpdatePlayerDTO): Promise<Player> {
    const response = await apiClient.patch<Player>(`/players/${id}`, data)
    return response.data
  },

  /**
   * Delete a player
   */
  async deletePlayer(id: number): Promise<void> {
    await apiClient.delete(`/players/${id}`)
  },
}
