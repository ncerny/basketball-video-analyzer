/**
 * API Client Service
 *
 * Centralized service for all backend API communication
 */

import type {
  Game,
  CreateGameDTO,
  UpdateGameDTO,
  Player,
  CreatePlayerDTO,
  UpdatePlayerDTO,
  Video,
  CreateVideoDTO,
  GameRoster,
  CreateGameRosterDTO,
  Annotation,
  CreateAnnotationDTO,
  UpdateAnnotationDTO,
  Play,
  CreatePlayDTO,
  APIError,
  ErrorResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      let errorData: ErrorResponse;
      try {
        errorData = await response.json();
      } catch {
        errorData = {
          message: `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      throw new APIError(
        errorData.message || 'An error occurred',
        response.status,
        errorData.code,
        errorData.details
      );
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    throw new APIError(
      error instanceof Error ? error.message : 'Network error',
      undefined,
      'NETWORK_ERROR'
    );
  }
}

/**
 * Games API
 */
export const gamesAPI = {
  /**
   * Get all games
   */
  list: () => fetchAPI<Game[]>('/api/games'),

  /**
   * Get a specific game by ID
   */
  get: (id: number) => fetchAPI<Game>(`/api/games/${id}`),

  /**
   * Create a new game
   */
  create: (data: CreateGameDTO) =>
    fetchAPI<Game>('/api/games', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update a game
   */
  update: (id: number, data: UpdateGameDTO) =>
    fetchAPI<Game>(`/api/games/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a game
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/games/${id}`, {
      method: 'DELETE',
    }),
};

/**
 * Players API
 */
export const playersAPI = {
  /**
   * Get all players
   */
  list: () => fetchAPI<Player[]>('/api/players'),

  /**
   * Get a specific player by ID
   */
  get: (id: number) => fetchAPI<Player>(`/api/players/${id}`),

  /**
   * Create a new player
   */
  create: (data: CreatePlayerDTO) =>
    fetchAPI<Player>('/api/players', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update a player
   */
  update: (id: number, data: UpdatePlayerDTO) =>
    fetchAPI<Player>(`/api/players/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a player
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/players/${id}`, {
      method: 'DELETE',
    }),
};

/**
 * Videos API
 */
export const videosAPI = {
  /**
   * Get all videos for a game
   */
  listByGame: (gameId: number) =>
    fetchAPI<Video[]>(`/api/videos?game_id=${gameId}`),

  /**
   * Get a specific video by ID
   */
  get: (id: number) => fetchAPI<Video>(`/api/videos/${id}`),

  /**
   * Upload a new video
   */
  upload: async (gameId: number, file: File): Promise<Video> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('game_id', gameId.toString());

    const response = await fetch(`${API_BASE_URL}/api/videos/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json();
      throw new APIError(
        errorData.message || 'Upload failed',
        response.status,
        errorData.code,
        errorData.details
      );
    }

    return await response.json();
  },

  /**
   * Update video metadata
   */
  update: (id: number, data: Partial<Video>) =>
    fetchAPI<Video>(`/api/videos/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a video
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/videos/${id}`, {
      method: 'DELETE',
    }),
};

/**
 * Game Rosters API
 */
export const gameRostersAPI = {
  /**
   * Get all roster entries for a game
   */
  listByGame: (gameId: number) =>
    fetchAPI<GameRoster[]>(`/api/game-rosters?game_id=${gameId}`),

  /**
   * Get a specific roster entry by ID
   */
  get: (id: number) => fetchAPI<GameRoster>(`/api/game-rosters/${id}`),

  /**
   * Add a player to a game roster
   */
  create: (data: CreateGameRosterDTO) =>
    fetchAPI<GameRoster>('/api/game-rosters', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update a roster entry
   */
  update: (id: number, data: Partial<CreateGameRosterDTO>) =>
    fetchAPI<GameRoster>(`/api/game-rosters/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Remove a player from a game roster
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/game-rosters/${id}`, {
      method: 'DELETE',
    }),
};

/**
 * Annotations API
 */
export const annotationsAPI = {
  /**
   * Get all annotations for a game
   */
  listByGame: (gameId: number) =>
    fetchAPI<Annotation[]>(`/api/annotations?game_id=${gameId}`),

  /**
   * Get a specific annotation by ID
   */
  get: (id: number) => fetchAPI<Annotation>(`/api/annotations/${id}`),

  /**
   * Create a new annotation
   */
  create: (data: CreateAnnotationDTO) =>
    fetchAPI<Annotation>('/api/annotations', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update an annotation
   */
  update: (id: number, data: UpdateAnnotationDTO) =>
    fetchAPI<Annotation>(`/api/annotations/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Delete an annotation
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/annotations/${id}`, {
      method: 'DELETE',
    }),

  /**
   * Verify an annotation (mark as verified)
   */
  verify: (id: number) =>
    fetchAPI<Annotation>(`/api/annotations/${id}/verify`, {
      method: 'POST',
    }),
};

/**
 * Plays API
 */
export const playsAPI = {
  /**
   * Get all plays for an annotation
   */
  listByAnnotation: (annotationId: number) =>
    fetchAPI<Play[]>(`/api/plays?annotation_id=${annotationId}`),

  /**
   * Get a specific play by ID
   */
  get: (id: number) => fetchAPI<Play>(`/api/plays/${id}`),

  /**
   * Create a new play
   */
  create: (data: CreatePlayDTO) =>
    fetchAPI<Play>('/api/plays', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  /**
   * Update a play
   */
  update: (id: number, data: Partial<CreatePlayDTO>) =>
    fetchAPI<Play>(`/api/plays/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  /**
   * Delete a play
   */
  delete: (id: number) =>
    fetchAPI<void>(`/api/plays/${id}`, {
      method: 'DELETE',
    }),
};

/**
 * Combined API client export
 */
export const api = {
  games: gamesAPI,
  players: playersAPI,
  videos: videosAPI,
  gameRosters: gameRostersAPI,
  annotations: annotationsAPI,
  plays: playsAPI,
};

export default api;
