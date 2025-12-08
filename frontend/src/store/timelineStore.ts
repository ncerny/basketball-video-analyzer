/**
 * Unified game timeline state management using Zustand
 *
 * This store manages the timeline across multiple videos, handling:
 * - Game time tracking (unified timeline across all videos)
 * - Video sequencing and transitions
 * - Playback control
 * - Annotations visualization
 */

import { create } from 'zustand';
import type { Video, Annotation, VideoTimeResult } from '../types/timeline';

interface TimelineState {
  // State
  /** Current game time in seconds (unified across all videos) */
  currentGameTime: number;

  /** ID of the currently playing video, or null if none */
  currentVideoId: number | null;

  /** Array of videos in the game timeline, sorted by sequence_order */
  videos: Video[];

  /** Whether playback is currently active */
  isPlaying: boolean;

  /** Playback speed multiplier (1.0 = normal speed) */
  playbackRate: number;

  /** Annotations for timeline visualization */
  annotations: Annotation[];

  // Actions
  /**
   * Update game time and automatically determine which video should play
   * @param time - Game time in seconds
   */
  setGameTime: (time: number) => void;

  /** Start playback */
  play: () => void;

  /** Pause playback */
  pause: () => void;

  /**
   * Change playback speed
   * @param rate - Playback rate (0.25, 0.5, 1.0, 2.0, etc.)
   */
  setPlaybackRate: (rate: number) => void;

  /**
   * Load video sequence for a game
   * @param videos - Array of videos with timeline metadata
   */
  loadVideos: (videos: Video[]) => void;

  /**
   * Jump to a specific game timestamp
   * @param time - Game time in seconds to seek to
   */
  seekToGameTime: (time: number) => void;

  /**
   * Calculate the current video-specific timestamp from game time
   * @returns Object containing the active video and its timestamp
   */
  getCurrentVideoTime: () => VideoTimeResult;

  /**
   * Convert video-specific time to game time
   * @param videoId - ID of the video
   * @param videoTime - Time within the video in seconds
   * @returns Game time in seconds
   */
  getGameTimeFromVideo: (videoId: number, videoTime: number) => number;

  /**
   * Load annotations for the current game
   * @param annotations - Array of annotations
   */
  loadAnnotations: (annotations: Annotation[]) => void;

  /**
   * Reset the store to initial state
   */
  reset: () => void;
}

/**
 * Find which video contains a specific game time
 * @param gameTime - Game time in seconds
 * @param videos - Sorted array of videos
 * @returns The video containing this game time, or null if in a gap
 */
const findVideoAtGameTime = (gameTime: number, videos: Video[]): Video | null => {
  if (videos.length === 0) return null;

  for (const video of videos) {
    if (video.game_time_offset == null) continue;

    const videoStart = video.game_time_offset;
    const videoEnd = video.game_time_offset + video.duration_seconds;

    if (gameTime >= videoStart && gameTime < videoEnd) {
      return video;
    }
  }

  return null;
};

/**
 * Convert game time to video-specific time
 * @param gameTime - Game time in seconds
 * @param video - The video to convert to
 * @returns Video-specific timestamp in seconds
 */
const gameTimeToVideoTime = (gameTime: number, video: Video): number => {
  return gameTime - (video.game_time_offset ?? 0);
};

/**
 * Convert video-specific time to game time
 * @param videoTime - Time within the video in seconds
 * @param video - The video
 * @returns Game time in seconds
 */
const videoTimeToGameTime = (videoTime: number, video: Video): number => {
  return (video.game_time_offset ?? 0) + videoTime;
};

/**
 * Sort videos by sequence order
 * @param videos - Array of videos to sort
 * @returns Sorted array (does not mutate original)
 */
const sortVideosBySequence = (videos: Video[]): Video[] => {
  return [...videos].sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0));
};

/**
 * Initial state for the timeline store
 */
const initialState = {
  currentGameTime: 0,
  currentVideoId: null,
  videos: [],
  isPlaying: false,
  playbackRate: 1.0,
  annotations: [],
};

/**
 * Zustand store for unified game timeline state
 */
export const useTimelineStore = create<TimelineState>((set, get) => ({
  ...initialState,

  setGameTime: (time: number) => {
    const { videos } = get();
    const activeVideo = findVideoAtGameTime(time, videos);

    set({
      currentGameTime: time,
      currentVideoId: activeVideo?.id || null,
    });
  },

  play: () => {
    set({ isPlaying: true });
  },

  pause: () => {
    set({ isPlaying: false });
  },

  setPlaybackRate: (rate: number) => {
    if (rate <= 0) {
      console.warn('Playback rate must be positive');
      return;
    }
    set({ playbackRate: rate });
  },

  loadVideos: (videos: Video[]) => {
    const sortedVideos = sortVideosBySequence(videos);

    // Reset game time to 0 and determine initial video
    const initialVideo = sortedVideos.length > 0 ? sortedVideos[0] : null;

    set({
      videos: sortedVideos,
      currentGameTime: 0,
      currentVideoId: initialVideo?.id || null,
      isPlaying: false,
    });
  },

  seekToGameTime: (time: number) => {
    const { videos } = get();

    // Clamp time to valid range
    const maxGameTime = videos.reduce((max, video) => {
      if (video.game_time_offset == null) return max;
      const videoEnd = video.game_time_offset + video.duration_seconds;
      return Math.max(max, videoEnd);
    }, 0);

    const clampedTime = Math.max(0, Math.min(time, maxGameTime));
    const activeVideo = findVideoAtGameTime(clampedTime, videos);

    set({
      currentGameTime: clampedTime,
      currentVideoId: activeVideo?.id || null,
    });
  },

  getCurrentVideoTime: (): VideoTimeResult => {
    const { currentGameTime, videos } = get();
    const video = findVideoAtGameTime(currentGameTime, videos);

    if (!video) {
      return {
        video: null,
        videoTime: 0,
        isInGap: true,
      };
    }

    return {
      video,
      videoTime: gameTimeToVideoTime(currentGameTime, video),
      isInGap: false,
    };
  },

  getGameTimeFromVideo: (videoId: number, videoTime: number): number => {
    const { videos } = get();
    const video = videos.find(v => v.id === videoId);

    if (!video) {
      console.warn(`Video with id ${videoId} not found`);
      return 0;
    }

    return videoTimeToGameTime(videoTime, video);
  },

  loadAnnotations: (annotations: Annotation[]) => {
    set({ annotations });
  },

  reset: () => {
    set(initialState);
  },
}));

/**
 * Helper hook to get video segments for timeline visualization
 * Returns array of segments showing which videos cover which time ranges
 */
export const useVideoSegments = () => {
  const videos = useTimelineStore(state => state.videos);

  return videos.map(video => ({
    videoId: video.id,
    start: video.game_time_offset ?? 0,
    end: (video.game_time_offset ?? 0) + video.duration_seconds,
    duration: video.duration_seconds,
  }));
};

/**
 * Helper hook to get gaps in video coverage
 * Returns array of time ranges where no video exists
 */
export const useVideoGaps = () => {
  const videos = useTimelineStore(state => state.videos);

  if (videos.length === 0) return [];

  const gaps: { start: number; end: number }[] = [];

  for (let i = 0; i < videos.length - 1; i++) {
    const currentVideo = videos[i];
    const nextVideo = videos[i + 1];

    if (currentVideo.game_time_offset == null || nextVideo.game_time_offset == null) continue;

    const currentVideoEnd = currentVideo.game_time_offset + currentVideo.duration_seconds;
    const nextVideoStart = nextVideo.game_time_offset;

    if (nextVideoStart > currentVideoEnd) {
      gaps.push({
        start: currentVideoEnd,
        end: nextVideoStart,
      });
    }
  }

  return gaps;
};

/**
 * Helper hook to detect overlapping video coverage
 * Returns array of time ranges where multiple videos exist
 */
export const useVideoOverlaps = () => {
  const videos = useTimelineStore(state => state.videos);

  if (videos.length === 0) return [];

  const overlaps: { start: number; end: number; videoIds: number[] }[] = [];

  for (let i = 0; i < videos.length - 1; i++) {
    const currentVideo = videos[i];
    const nextVideo = videos[i + 1];

    if (currentVideo.game_time_offset == null || nextVideo.game_time_offset == null) continue;

    const currentVideoEnd = currentVideo.game_time_offset + currentVideo.duration_seconds;
    const nextVideoStart = nextVideo.game_time_offset;

    // Check if current video overlaps with next
    if (nextVideoStart < currentVideoEnd) {
      const nextVideoEnd = nextVideo.game_time_offset + nextVideo.duration_seconds;

      overlaps.push({
        start: nextVideoStart,
        end: Math.min(currentVideoEnd, nextVideoEnd),
        videoIds: [currentVideo.id, nextVideo.id],
      });
    }
  }

  return overlaps;
};

export default useTimelineStore;
