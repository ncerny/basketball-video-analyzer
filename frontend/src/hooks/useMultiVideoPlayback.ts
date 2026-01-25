/**
 * Multi-video playback engine hook
 *
 * Manages multiple video elements with:
 * - Seamless transitions between videos
 * - Pre-buffering of next video
 * - Game time to video time conversion
 * - Synchronization with timeline store
 * - Memory-efficient video management
 */

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import { videosAPI } from '../api/videos';
import type { Video } from '../types/timeline';

interface VideoElement {
  video: HTMLVideoElement;
  videoData: Video;
  isPreBuffered: boolean;
  lastUsedTime: number; // Track when video was last used for memory management
  presignedUrl?: string; // Cached presigned URL for R2 videos
  presignedUrlExpiresAt?: number; // Timestamp when presigned URL expires
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

  /** Maximum number of video elements to keep in memory */
  maxVideoElements?: number;
  
  /** Maximum time to display a gap (longer gaps will be accelerated) */
  maxGapDisplayTime?: number;
}

interface GapDisplayInfo {
  /** Start of the gap in game time */
  gapStart: number;
  /** End of the gap in game time */
  gapEnd: number;
  /** Actual duration of the gap in game time */
  actualGapDuration: number;
  /** Display duration (capped at maxGapDisplayTime) */
  displayDuration: number;
  /** Time elapsed in display time (0 to displayDuration) */
  displayElapsed: number;
  /** Time remaining in display time */
  displayRemaining: number;
}

interface UseMultiVideoPlaybackReturn {
  /** Current video element to display */
  currentVideoElement: HTMLVideoElement | null;

  /** Whether currently buffering */
  isBuffering: boolean;

  /** Whether in a gap between videos */
  isInGap: boolean;
  
  /** Gap display information for UI (countdown, etc.) */
  gapDisplayInfo: GapDisplayInfo | null;

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
    preBufferThreshold = 5, // Start buffering 5 seconds before transition (increased from 3)
    maxVideoElements = 10, // Keep at most 10 video elements in memory (increased to avoid evicting needed videos)
    maxGapDisplayTime = 10, // Maximum seconds to display a gap (longer gaps are accelerated)
  } = options;

  // Timeline store state
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const currentVideoId = useTimelineStore(state => state.currentVideoId);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const playbackRate = useTimelineStore(state => state.playbackRate);
  const videos = useTimelineStore(state => state.videos);
  const getCurrentVideoTime = useTimelineStore(state => state.getCurrentVideoTime);
  const setGameTime = useTimelineStore(state => state.setGameTime);
  const totalTimelineDuration = useMemo(() => {
    return videos.reduce((max, video) => {
      if (video.game_time_offset == null) return max;
      return Math.max(max, video.game_time_offset + video.duration_seconds);
    }, 0);
  }, [videos]);

  // Video element management
  const videoElementsRef = useRef<Map<number, VideoElement>>(new Map());
  const [currentVideoElement, setCurrentVideoElement] = useState<HTMLVideoElement | null>(null);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isInGap, setIsInGap] = useState(false);
  const [gapDisplayInfo, setGapDisplayInfo] = useState<GapDisplayInfo | null>(null);
  const lastFrameTimeRef = useRef<number | null>(null);
  const gapTrackerRef = useRef<{ 
    end: number; 
    anchorTime: number; 
    startedAt: number;
    actualDuration: number;
    displayDuration: number;
  } | null>(null);

  // Animation frame for game time updates
  const animationFrameRef = useRef<number | undefined>(undefined);
  const isSyncingRef = useRef<boolean>(false); // Prevent feedback loops
  
  // Refs to avoid stale closures in RAF callback - sync immediately during render
  const currentGameTimeRef = useRef(currentGameTime);
  const playbackRateRef = useRef(playbackRate);
  const isPlayingRef = useRef(isPlaying);
  
  // Update refs synchronously (safe during render for refs)
  currentGameTimeRef.current = currentGameTime;
  playbackRateRef.current = playbackRate;
  isPlayingRef.current = isPlaying;

  const getNextVideoStart = useCallback((time: number) => {
    // Sort by game_time_offset to find the next video correctly
    const sortedByOffset = [...videos]
      .filter(v => v.game_time_offset != null)
      .sort((a, b) => (a.game_time_offset ?? 0) - (b.game_time_offset ?? 0));
    
    for (const video of sortedByOffset) {
      if (video.game_time_offset! > time + 0.001) {
        return video.game_time_offset!;
      }
    }
    return null;
  }, [videos]);

  /**
   * Get the video source URL for a video
   * For R2 videos, fetches a presigned URL
   * For local videos, returns the /stream endpoint
   */
  const getVideoSourceUrl = useCallback(async (videoData: Video): Promise<{ url: string; expiresAt?: number }> => {
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    // For R2 videos, try to get a presigned URL
    if (videoData.r2_key) {
      try {
        const { url, expires_in } = await videosAPI.getStreamUrl(videoData.id);
        const expiresAt = Date.now() + (expires_in * 1000) - 60000; // Subtract 1 minute buffer
        return { url, expiresAt };
      } catch (error) {
        console.warn(`Failed to get presigned URL for video ${videoData.id}, falling back to stream endpoint:`, error);
        // Fall back to stream endpoint (which will redirect to presigned URL)
        return { url: `${API_BASE_URL}/api/videos/${videoData.id}/stream` };
      }
    }

    // For local videos, use the stream endpoint
    return { url: `${API_BASE_URL}/api/videos/${videoData.id}/stream` };
  }, []);

  /**
   * Refresh the presigned URL for a video element
   */
  const refreshPresignedUrl = useCallback(async (videoElement: VideoElement) => {
    if (!videoElement.videoData.r2_key) return;

    try {
      const { url, expiresAt } = await getVideoSourceUrl(videoElement.videoData);
      videoElement.presignedUrl = url;
      videoElement.presignedUrlExpiresAt = expiresAt;

      // Update the video source
      const currentTime = videoElement.video.currentTime;
      const wasPlaying = !videoElement.video.paused;

      videoElement.video.src = url;
      videoElement.video.load();

      // Restore playback position after source change
      videoElement.video.addEventListener('loadedmetadata', () => {
        videoElement.video.currentTime = currentTime;
        if (wasPlaying) {
          videoElement.video.play().catch(() => {});
        }
      }, { once: true });

      console.log(`Refreshed presigned URL for video ${videoElement.videoData.id}`);
    } catch (error) {
      console.error(`Failed to refresh presigned URL for video ${videoElement.videoData.id}:`, error);
    }
  }, [getVideoSourceUrl]);

  /**
   * Create a video element for a video file
   */
  const createVideoElement = useCallback(async (
    videoData: Video,
    callbacks: { onBuffering?: () => void; onBuffered?: () => void }
  ): Promise<{ element: HTMLVideoElement; presignedUrl?: string; expiresAt?: number }> => {
    const video = document.createElement('video');
    video.preload = 'auto'; // Preload the entire video for smoother playback
    video.playsInline = true;
    video.muted = false;

    // Get the video source URL (presigned for R2, /stream for local)
    const { url, expiresAt } = await getVideoSourceUrl(videoData);
    video.src = url;

    // Add event listeners for buffering detection
    video.addEventListener('waiting', () => {
      setIsBuffering(true);
      callbacks.onBuffering?.();
    });

    video.addEventListener('canplay', () => {
      setIsBuffering(false);
      callbacks.onBuffered?.();
    });

    // Also listen for canplaythrough - video can play without stopping
    video.addEventListener('canplaythrough', () => {
      setIsBuffering(false);
      callbacks.onBuffered?.();
    });

    // Handle 403 errors (expired presigned URL) by refreshing
    video.addEventListener('error', async () => {
      const error = video.error;
      // MediaError code 4 is MEDIA_ERR_SRC_NOT_SUPPORTED, which includes 403
      if (error && error.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED && videoData.r2_key) {
        console.warn(`Video source error for ${videoData.id}, attempting to refresh presigned URL`);
        const videoEl = videoElementsRef.current.get(videoData.id);
        if (videoEl) {
          await refreshPresignedUrl(videoEl);
        }
      }
    });

    return { element: video, presignedUrl: videoData.r2_key ? url : undefined, expiresAt };
  }, [getVideoSourceUrl, refreshPresignedUrl]);

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

    // Create video elements for all videos (in parallel for efficiency)
    const createPromises = videosToLoad.map(async (videoData) => {
      const { element, presignedUrl, expiresAt } = await createVideoElement(videoData, { onBuffering, onBuffered });
      return {
        videoData,
        element,
        presignedUrl,
        expiresAt,
      };
    });

    const results = await Promise.all(createPromises);

    for (const { videoData, element, presignedUrl, expiresAt } of results) {
      videoElementsRef.current.set(videoData.id, {
        video: element,
        videoData,
        isPreBuffered: false,
        lastUsedTime: Date.now(),
        presignedUrl,
        presignedUrlExpiresAt: expiresAt,
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
        firstVideo.lastUsedTime = Date.now();
      }
    }
  }, [createVideoElement, onBuffering, onBuffered]);

  /**
   * Evict least recently used video elements to save memory
   * Keeps the current video and at most maxVideoElements total
   */
  const evictOldVideoElements = useCallback((currentVideoId: number | null) => {
    if (videoElementsRef.current.size <= maxVideoElements) return;

    // Find current video index and determine which videos to protect
    const currentIndex = videos.findIndex(v => v.id === currentVideoId);
    const protectedIds = new Set<number>();
    
    // Protect current video and adjacent videos (prev and next)
    if (currentVideoId !== null) protectedIds.add(currentVideoId);
    if (currentIndex > 0) protectedIds.add(videos[currentIndex - 1].id);
    if (currentIndex >= 0 && currentIndex < videos.length - 1) protectedIds.add(videos[currentIndex + 1].id);
    
    // Get all video elements sorted by last used time (oldest first), excluding protected
    const entries = Array.from(videoElementsRef.current.entries())
      .filter(([id]) => !protectedIds.has(id))
      .sort((a, b) => a[1].lastUsedTime - b[1].lastUsedTime);

    // Evict oldest videos until we're at the limit
    const toEvict = entries.slice(0, videoElementsRef.current.size - maxVideoElements);
    for (const [id, element] of toEvict) {
      element.video.pause();
      element.video.src = '';
      element.video.load();
      videoElementsRef.current.delete(id);
    }
  }, [maxVideoElements, videos]);

  /**
   * Create a video element synchronously (for evicted videos)
   * Uses /stream endpoint which handles R2 redirects
   */
  const createVideoElementSync = useCallback((
    videoData: Video,
    callbacks: { onBuffering?: () => void; onBuffered?: () => void }
  ): HTMLVideoElement => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.playsInline = true;
    video.muted = false;

    // Use /stream endpoint (handles both local and R2 with redirect)
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    video.src = `${API_BASE_URL}/api/videos/${videoData.id}/stream`;

    video.addEventListener('waiting', () => {
      setIsBuffering(true);
      callbacks.onBuffering?.();
    });

    video.addEventListener('canplay', () => {
      setIsBuffering(false);
      callbacks.onBuffered?.();
    });

    video.addEventListener('canplaythrough', () => {
      setIsBuffering(false);
      callbacks.onBuffered?.();
    });

    // Handle errors by attempting to refresh presigned URL for R2 videos
    video.addEventListener('error', async () => {
      const error = video.error;
      if (error && error.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED && videoData.r2_key) {
        const videoEl = videoElementsRef.current.get(videoData.id);
        if (videoEl) {
          await refreshPresignedUrl(videoEl);
        }
      }
    });

    return video;
  }, [refreshPresignedUrl]);

  const ensureVideoElement = useCallback((videoData: Video): VideoElement | null => {
    let entry = videoElementsRef.current.get(videoData.id);
    if (!entry) {
      // Video was evicted or never created - recreate it synchronously
      const video = createVideoElementSync(videoData, { onBuffering, onBuffered });
      entry = {
        video,
        videoData,
        isPreBuffered: false,
        lastUsedTime: Date.now(),
      };
      videoElementsRef.current.set(videoData.id, entry);
      video.load();
    }
    return entry;
  }, [createVideoElementSync, onBuffering, onBuffered]);

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
    // Skip sync if we're updating game time from video playback (prevents feedback loop)
    if (isSyncingRef.current) return;

    const { video: timelineVideo, videoTime, isInGap: inGap } = getCurrentVideoTime();

    setIsInGap(inGap);

    if (!timelineVideo) {
      // In a gap - pause current video
      if (currentVideoElement) {
        currentVideoElement.pause();
      }

      // Set up gap tracker to advance time through the gap
      const nextStart = getNextVideoStart(currentGameTime);
      const gapEnd = nextStart ?? totalTimelineDuration;
      
      // Only set up tracker if we have somewhere to go
      if (gapEnd > currentGameTime + 0.001) {
        // Check if we need to reset the tracker (new gap or significant drift)
        const existingTracker = gapTrackerRef.current;
        const shouldResetTracker =
          !existingTracker ||
          Math.abs(existingTracker.end - gapEnd) > 0.1;

        if (shouldResetTracker) {
          const actualDuration = gapEnd - currentGameTime;
          const displayDuration = Math.min(actualDuration, maxGapDisplayTime);
          gapTrackerRef.current = {
            end: gapEnd,
            anchorTime: currentGameTime,
            startedAt: performance.now(),
            actualDuration,
            displayDuration,
          };
        }
      } else {
        // We've reached the end of the timeline
        gapTrackerRef.current = null;
        setGapDisplayInfo(null);
      }
      return;
    }

    // Exiting a gap - clear the tracker
    if (gapTrackerRef.current) {
      setGapDisplayInfo(null);
    }
    gapTrackerRef.current = null;

    // Get or switch to the correct video element (ensure it exists)
    const videoElement = ensureVideoElement(timelineVideo);
    if (!videoElement) return;

    // Update last used time for memory management
    videoElement.lastUsedTime = Date.now();

    // Switch video element if changed
    if (currentVideoElement !== videoElement.video) {
      currentVideoElement?.pause();
      setCurrentVideoElement(videoElement.video);
      onVideoChange?.(timelineVideo);
      
      // Evict old video elements to save memory
      evictOldVideoElements(timelineVideo.id);
    }

    // Sync video time only when difference is significant (0.5s tolerance to reduce seeking)
    // This prevents constant micro-seeking which causes stuttering
    const timeDiff = Math.abs(videoElement.video.currentTime - videoTime);
    if (timeDiff > 0.5) {
      videoElement.video.currentTime = videoTime;
    }

    // Sync playback state - but only if video is ready
    const isVideoReady = videoElement.video.readyState >= 3; // HAVE_FUTURE_DATA or better
    
    if (isPlaying && videoElement.video.paused && isVideoReady) {
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
  }, [getCurrentVideoTime, currentVideoElement, isPlaying, playbackRate, onVideoChange, checkPreBuffer, evictOldVideoElements, currentGameTime, getNextVideoStart, totalTimelineDuration, ensureVideoElement, maxGapDisplayTime]);

  /**
   * Update game time based on current video playback
   * Uses the video's currentTime as the source of truth to avoid sync issues
   */
  const updateGameTime = useCallback((timestamp: number) => {
    if (!isPlayingRef.current) {
      lastFrameTimeRef.current = null;
      // Don't schedule next frame when paused
      return;
    }

    // Always schedule next frame first to ensure loop continues
    animationFrameRef.current = requestAnimationFrame(updateGameTime);

    if (lastFrameTimeRef.current == null) {
      lastFrameTimeRef.current = timestamp;
    }

    const rate = playbackRateRef.current;
    const gameTime = currentGameTimeRef.current;
    const deltaSeconds = ((timestamp - lastFrameTimeRef.current) / 1000) * rate;
    lastFrameTimeRef.current = timestamp;

    // Check current state
    const { video: timelineVideo, isInGap: timelineInGap } = getCurrentVideoTime();
    let advancedViaVideo = false;
    let advancedViaGap = false;

    // Handle gap traversal - set up tracker if needed
    if (timelineInGap && !gapTrackerRef.current) {
      // We're in a gap but don't have a tracker - set one up
      const sortedByOffset = [...videos]
        .filter(v => v.game_time_offset != null)
        .sort((a, b) => (a.game_time_offset ?? 0) - (b.game_time_offset ?? 0));
      
      let nextStart: number | null = null;
      for (const video of sortedByOffset) {
        if (video.game_time_offset! > gameTime + 0.001) {
          nextStart = video.game_time_offset!;
          break;
        }
      }
      
      const gapEnd = nextStart ?? totalTimelineDuration;
      if (gapEnd > gameTime + 0.001) {
        const actualDuration = gapEnd - gameTime;
        const displayDuration = Math.min(actualDuration, maxGapDisplayTime);
        gapTrackerRef.current = {
          end: gapEnd,
          anchorTime: gameTime,
          startedAt: timestamp,
          actualDuration,
          displayDuration,
        };
      }
    }

    // Handle gap traversal
    const gapTracker = gapTrackerRef.current;
    if (gapTracker && timelineInGap) {
      // Calculate display elapsed time (real time since gap started)
      const displayElapsed = ((timestamp - gapTracker.startedAt) / 1000) * rate;
      
      // Calculate the speed multiplier to traverse the gap in displayDuration time
      const speedMultiplier = gapTracker.actualDuration / gapTracker.displayDuration;
      
      // Calculate actual game time progress (accelerated for long gaps)
      const gameTimeProgress = displayElapsed * speedMultiplier;
      const targetTime = Math.min(gapTracker.anchorTime + gameTimeProgress, gapTracker.end);
      
      // Update gap display info for UI
      const displayRemaining = Math.max(0, gapTracker.displayDuration - displayElapsed);
      setGapDisplayInfo({
        gapStart: gapTracker.anchorTime,
        gapEnd: gapTracker.end,
        actualGapDuration: gapTracker.actualDuration,
        displayDuration: gapTracker.displayDuration,
        displayElapsed: Math.min(displayElapsed, gapTracker.displayDuration),
        displayRemaining,
      });

      if (targetTime > gameTime + 0.0005) {
        isSyncingRef.current = true;
        setGameTime(targetTime);
        isSyncingRef.current = false;
      }

      if (targetTime >= gapTracker.end - 0.0005 || displayElapsed >= gapTracker.displayDuration) {
        gapTrackerRef.current = null;
        setGapDisplayInfo(null);
        // Jump to gap end if we haven't reached it yet
        if (targetTime < gapTracker.end - 0.01) {
          isSyncingRef.current = true;
          setGameTime(gapTracker.end);
          isSyncingRef.current = false;
        }
      }

      advancedViaGap = true;
    }

    if (!timelineInGap && timelineVideo) {
      // Clear gap tracker when we exit a gap
      if (gapTrackerRef.current) {
        gapTrackerRef.current = null;
        setGapDisplayInfo(null);
      }
      
      // Ensure video element exists (may have been evicted)
      const entry = ensureVideoElement(timelineVideo);
      if (!entry) return;
      
      if (entry.videoData.game_time_offset != null) {
        const element = entry.video;
        
        // Check if video has ended (reached its duration)
        const videoEnded = element.ended || 
          (element.duration > 0 && element.currentTime >= element.duration - 0.1);
        
        if (videoEnded) {
          // Video has ended - jump to end of this video's timeline coverage
          const videoEndGameTime = entry.videoData.game_time_offset + entry.videoData.duration_seconds;
          isSyncingRef.current = true;
          setGameTime(videoEndGameTime + 0.001); // Small offset to enter the gap
          isSyncingRef.current = false;
          advancedViaVideo = true;
        } else {
          const newGameTime = entry.videoData.game_time_offset + element.currentTime;

          if (Math.abs(newGameTime - gameTime) > 0.05) {
            isSyncingRef.current = true;
            setGameTime(newGameTime);
            isSyncingRef.current = false;
          }
          advancedViaVideo = true;
        }
      }
    }

    // Fallback: if neither video nor gap advanced time, use delta
    if (!advancedViaVideo && !advancedViaGap && deltaSeconds > 0) {
      isSyncingRef.current = true;
      setGameTime(gameTime + deltaSeconds);
      isSyncingRef.current = false;
    }
  }, [getCurrentVideoTime, setGameTime, videos, totalTimelineDuration, ensureVideoElement, maxGapDisplayTime]);

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
      if (gapTrackerRef.current) {
        gapTrackerRef.current.anchorTime = currentGameTime;
        gapTrackerRef.current.startedAt = performance.now();
      }
      animationFrameRef.current = requestAnimationFrame(updateGameTime);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      lastFrameTimeRef.current = null;
      if (gapTrackerRef.current) {
        gapTrackerRef.current.anchorTime = currentGameTime;
        gapTrackerRef.current.startedAt = performance.now();
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
    gapDisplayInfo,
    loadVideoFiles,
    cleanup,
  };
};

export default useMultiVideoPlayback;
