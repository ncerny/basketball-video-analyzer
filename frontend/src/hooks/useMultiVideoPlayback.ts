/**
 * Multi-video playback engine hook
 *
 * Manages multiple video elements with:
 * - Seamless transitions between videos
 * - Pre-buffering of next video
 * - Game time to video time conversion
 * - Synchronization with timeline store
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import type { Video } from '../types/timeline';

interface VideoElement {
  video: HTMLVideoElement;
  videoData: Video;
  isPreBuffered: boolean;
}

interface UseMultiVideoPlaybackOptions {
  /** Callback when video transition occurs */
  onVideoChange?: (video: Video | null) => void;

  /** Callback when buffering starts */
  onBuffering?: () => void;

  /** Callback when buffering completes */
  onBuffered?: () => void;

  /** How far ahead to start pre-buffering next video (seconds) */
  preBufferThreshold?: number;
}

interface UseMultiVideoPlaybackReturn {
  /** Current video element to display */
  currentVideoElement: HTMLVideoElement | null;

  /** Whether currently buffering */
  isBuffering: boolean;

  /** Whether in a gap between videos */
  isInGap: boolean;

  /** Load video files */
  loadVideoFiles: (videos: Video[]) => Promise<void>;

  /** Clean up all video elements */
  cleanup: () => void;
}

/**
 * Hook to manage multi-video playback with smooth transitions
 */
export const useMultiVideoPlayback = (
  options: UseMultiVideoPlaybackOptions = {}
): UseMultiVideoPlaybackReturn => {
  const {
    onVideoChange,
    onBuffering,
    onBuffered,
    preBufferThreshold = 3, // Start buffering 3 seconds before transition
  } = options;

  // Timeline store state
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const currentVideoId = useTimelineStore(state => state.currentVideoId);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const playbackRate = useTimelineStore(state => state.playbackRate);
  const videos = useTimelineStore(state => state.videos);
  const getCurrentVideoTime = useTimelineStore(state => state.getCurrentVideoTime);
  const setGameTime = useTimelineStore(state => state.setGameTime);

  // Video element management
  const videoElementsRef = useRef<Map<number, VideoElement>>(new Map());
  const [currentVideoElement, setCurrentVideoElement] = useState<HTMLVideoElement | null>(null);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isInGap, setIsInGap] = useState(false);

  // Animation frame for game time updates
  const animationFrameRef = useRef<number | undefined>(undefined);
  const lastUpdateTimeRef = useRef<number>(Date.now());

  /**
   * Create a video element for a video file
   */
  const createVideoElement = useCallback((videoData: Video, callbacks: { onBuffering?: () => void; onBuffered?: () => void }): HTMLVideoElement => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.playsInline = true;

    // Set video source - in production this would use actual video URLs
    // For now, using file_path directly
    video.src = videoData.file_path;

    // Add event listeners for buffering detection
    video.addEventListener('waiting', () => {
      setIsBuffering(true);
      callbacks.onBuffering?.();
    });

    video.addEventListener('canplay', () => {
      setIsBuffering(false);
      callbacks.onBuffered?.();
    });

    return video;
  }, []);

  /**
   * Load video files and create video elements
   */
  const loadVideoFiles = useCallback(async (videosToLoad: Video[]) => {
    // Clean up existing video elements
    videoElementsRef.current.forEach(({ video }) => {
      video.pause();
      video.src = '';
      video.load();
    });
    videoElementsRef.current.clear();

    // Create video elements for all videos
    for (const videoData of videosToLoad) {
      const video = createVideoElement(videoData, { onBuffering, onBuffered });
      videoElementsRef.current.set(videoData.id, {
        video,
        videoData,
        isPreBuffered: false,
      });
    }

    // Load metadata for the first video
    if (videosToLoad.length > 0) {
      const firstVideo = videoElementsRef.current.get(videosToLoad[0].id);
      if (firstVideo) {
        await new Promise<void>((resolve) => {
          firstVideo.video.addEventListener('loadedmetadata', () => resolve(), { once: true });
          firstVideo.video.load();
        });
        setCurrentVideoElement(firstVideo.video);
        firstVideo.isPreBuffered = true;
      }
    }
  }, [createVideoElement, onBuffering, onBuffered]);

  /**
   * Pre-buffer the next video in sequence
   */
  const preBufferNextVideo = useCallback(async (currentVideo: Video) => {
    const currentIndex = videos.findIndex(v => v.id === currentVideo.id);
    if (currentIndex === -1 || currentIndex === videos.length - 1) return;

    const nextVideo = videos[currentIndex + 1];
    const nextElement = videoElementsRef.current.get(nextVideo.id);

    if (nextElement && !nextElement.isPreBuffered) {
      // Pre-load the next video
      nextElement.video.preload = 'auto';
      nextElement.video.load();
      nextElement.isPreBuffered = true;
    }
  }, [videos]);

  /**
   * Check if we should start pre-buffering
   */
  const checkPreBuffer = useCallback(() => {
    const { video, videoTime } = getCurrentVideoTime();
    if (!video) return;

    const timeUntilEnd = video.duration_seconds - videoTime;
    if (timeUntilEnd <= preBufferThreshold) {
      preBufferNextVideo(video);
    }
  }, [getCurrentVideoTime, preBufferThreshold, preBufferNextVideo]);

  /**
   * Sync video element with timeline store
   */
  const syncVideoElement = useCallback(() => {
    const { video: timelineVideo, videoTime, isInGap: inGap } = getCurrentVideoTime();

    setIsInGap(inGap);

    if (!timelineVideo) {
      // In a gap - pause current video
      if (currentVideoElement) {
        currentVideoElement.pause();
      }
      return;
    }

    // Get or switch to the correct video element
    const videoElement = videoElementsRef.current.get(timelineVideo.id);
    if (!videoElement) return;

    // Switch video element if changed
    if (currentVideoElement !== videoElement.video) {
      currentVideoElement?.pause();
      setCurrentVideoElement(videoElement.video);
      onVideoChange?.(timelineVideo);
    }

    // Sync video time (with small tolerance to avoid constant seeking)
    const timeDiff = Math.abs(videoElement.video.currentTime - videoTime);
    if (timeDiff > 0.1) {
      videoElement.video.currentTime = videoTime;
    }

    // Sync playback state
    if (isPlaying && videoElement.video.paused) {
      videoElement.video.play().catch(err => {
        console.warn('Failed to play video:', err);
      });
    } else if (!isPlaying && !videoElement.video.paused) {
      videoElement.video.pause();
    }

    // Sync playback rate
    if (videoElement.video.playbackRate !== playbackRate) {
      videoElement.video.playbackRate = playbackRate;
    }

    // Check if we should pre-buffer next video
    checkPreBuffer();
  }, [getCurrentVideoTime, currentVideoElement, isPlaying, playbackRate, onVideoChange, checkPreBuffer]);

  /**
   * Update game time based on current video playback
   */
  const updateGameTime = useCallback(() => {
    if (!isPlaying || !currentVideoElement) return;

    const now = Date.now();
    const deltaTime = (now - lastUpdateTimeRef.current) / 1000; // Convert to seconds
    lastUpdateTimeRef.current = now;

    // Update game time based on video playback
    const newGameTime = currentGameTime + (deltaTime * playbackRate);
    setGameTime(newGameTime);

    animationFrameRef.current = requestAnimationFrame(updateGameTime);
  }, [isPlaying, currentVideoElement, currentGameTime, playbackRate, setGameTime]);

  /**
   * Cleanup function
   */
  const cleanup = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    videoElementsRef.current.forEach(({ video }) => {
      video.pause();
      video.src = '';
      video.load();
    });
    videoElementsRef.current.clear();
    setCurrentVideoElement(null);
  }, []);

  // Sync video when timeline state changes
  useEffect(() => {
    syncVideoElement();
  }, [currentVideoId, currentGameTime, isPlaying, playbackRate, syncVideoElement]);

  // Start/stop game time updates when playing state changes
  useEffect(() => {
    if (isPlaying) {
      lastUpdateTimeRef.current = Date.now();
      animationFrameRef.current = requestAnimationFrame(updateGameTime);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, updateGameTime]);

  // Load videos when videos array changes
  useEffect(() => {
    if (videos.length > 0) {
      loadVideoFiles(videos);
    }
  }, [videos, loadVideoFiles]);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    currentVideoElement,
    isBuffering,
    isInGap,
    loadVideoFiles,
    cleanup,
  };
};

export default useMultiVideoPlayback;
