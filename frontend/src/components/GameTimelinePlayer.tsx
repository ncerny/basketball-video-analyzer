/**
 * Game Timeline Player UI Component
 *
 * Comprehensive video player with:
 * - Multi-video playback
 * - Current video progress bar with seeking
 * - Full playback controls
 * - Frame-by-frame navigation
 */

import { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import {
  ActionIcon,
  Box,
  Button,
  Group,
  Menu,
  Paper,
  Stack,
  Switch,
  Text,
  Tooltip,
} from '@mantine/core';
import {
  IconEye,
  IconEyeOff,
  IconGauge,
  IconPlayerPause,
  IconPlayerPlay,
  IconPlayerSkipBack,
  IconPlayerSkipForward,
  IconPlayerTrackNext,
  IconPlayerTrackPrev,
} from '@tabler/icons-react';
import { useTimelineStore } from '../store/timelineStore';
import { MultiVideoPlayer } from './MultiVideoPlayer';
import { VideoProgressBar } from './VideoProgressBar';
import { DetectionOverlay } from './DetectionOverlay';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { detectionAPI, jerseyNumbersAPI } from '../api';
import type { AggregatedJerseyNumber } from '../api/jerseyNumbers';
import type { Video } from '../types/timeline';
import type { Detection } from '../types/api';

// Windowed detection loading parameters
const WINDOW_SIZE = 1000;        // frames (~33 seconds at 30fps)
const BUFFER_THRESHOLD = 300;    // frames from edge to trigger prefetch

interface GameTimelinePlayerProps {
  /** CSS class name for the container */
  className?: string;

  /** Show advanced controls (frame-by-frame, etc.) */
  showAdvancedControls?: boolean;

  /** Callback when video changes */
  onVideoChange?: (video: Video | null) => void;
  
  /** Callback when in a gap */
  onGapChange?: (isInGap: boolean) => void;
}

/**
 * Game Timeline Player - Full-featured video player with timeline
 */
export const GameTimelinePlayer: React.FC<GameTimelinePlayerProps> = ({
  className = '',
  showAdvancedControls = true,
  onVideoChange,
  onGapChange,
}) => {
  // Timeline store state
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const playbackRate = useTimelineStore(state => state.playbackRate);
  const videos = useTimelineStore(state => state.videos);
  const getCurrentVideoTime = useTimelineStore(state => state.getCurrentVideoTime);

  // Timeline store actions
  const play = useTimelineStore(state => state.play);
  const pause = useTimelineStore(state => state.pause);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const setPlaybackRate = useTimelineStore(state => state.setPlaybackRate);

  // Local state for tracking current video and gap status
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [isInGap, setIsInGap] = useState(false);

  // Detection overlay state
  const [showDetections, setShowDetections] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [jerseyNumbers, setJerseyNumbers] = useState<Map<number, AggregatedJerseyNumber>>(new Map());
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  // Windowed detection loading state
  const [loadedRange, setLoadedRange] = useState<{start: number, end: number} | null>(null);
  const [_isLoadingDetections, setIsLoadingDetections] = useState(false); // For future UI indicator
  const loadingRef = useRef(false); // Prevent concurrent fetches

  // Fetch detections for a window around the target frame
  const fetchWindow = useCallback(async (
    videoId: number,
    targetFrame: number,
    mode: 'replace' | 'extend-forward' | 'extend-backward' = 'replace'
  ) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setIsLoadingDetections(true);

    try {
      let frameStart: number;
      let frameEnd: number;

      if (mode === 'replace') {
        // Center window on target frame
        frameStart = Math.max(0, targetFrame - Math.floor(WINDOW_SIZE / 2));
        frameEnd = targetFrame + Math.floor(WINDOW_SIZE / 2);
      } else if (mode === 'extend-forward') {
        // Fetch next window (for forward playback)
        frameStart = loadedRange?.end ?? targetFrame;
        frameEnd = frameStart + WINDOW_SIZE;
      } else {
        // Fetch previous window (for backward playback)
        frameEnd = loadedRange?.start ?? targetFrame;
        frameStart = Math.max(0, frameEnd - WINDOW_SIZE);
      }

      const response = await detectionAPI.getVideoDetections(videoId, {
        frame_start: frameStart,
        frame_end: frameEnd,
        limit: 50000, // ~1600 frames * 31 detections/frame = ~50k max
      });

      if (mode === 'replace') {
        setDetections(response.detections);
        setLoadedRange({ start: frameStart, end: frameEnd });
      } else if (mode === 'extend-forward') {
        // Merge and trim old detections
        const trimStart = targetFrame - WINDOW_SIZE;
        setDetections(prev => {
          const trimmed = prev.filter(d => d.frame_number >= trimStart);
          return [...trimmed, ...response.detections];
        });
        setLoadedRange(prev => ({
          start: Math.max(trimStart, prev?.start ?? 0),
          end: frameEnd,
        }));
      } else {
        // Merge backward and trim future detections
        const trimEnd = targetFrame + WINDOW_SIZE;
        setDetections(prev => {
          const trimmed = prev.filter(d => d.frame_number <= trimEnd);
          return [...response.detections, ...trimmed];
        });
        setLoadedRange(prev => ({
          start: frameStart,
          end: Math.min(trimEnd, prev?.end ?? frameEnd),
        }));
      }
    } catch (err) {
      console.error('Failed to load detections:', err);
      if (mode === 'replace') {
        setDetections([]);
        setLoadedRange(null);
      }
    } finally {
      loadingRef.current = false;
      setIsLoadingDetections(false);
    }
  }, [loadedRange]);

  // Handle video change from MultiVideoPlayer
  const handleVideoChange = useCallback((video: Video | null) => {
    setCurrentVideo(video);
    onVideoChange?.(video);

    // Reset detection state on video change
    setLoadedRange(null);

    // Load detections and jersey numbers when video changes
    if (video && showDetections) {
      // Windowed detection loading - start at frame 0
      fetchWindow(video.id, 0, 'replace');

      // Jersey numbers are small, load all at once
      jerseyNumbersAPI.getByTrack(video.id)
        .then(response => {
          const map = new Map<number, AggregatedJerseyNumber>();
          for (const track of response.tracks) {
            map.set(track.tracking_id, track);
          }
          setJerseyNumbers(map);
        })
        .catch(err => {
          console.error('Failed to load jersey numbers:', err);
          setJerseyNumbers(new Map());
        });
    } else {
      setDetections([]);
      setJerseyNumbers(new Map());
    }
  }, [onVideoChange, showDetections, fetchWindow]);

  // Check gap status from store
  const videoTimeResult = getCurrentVideoTime();
  if (videoTimeResult.isInGap !== isInGap) {
    setIsInGap(videoTimeResult.isInGap);
    onGapChange?.(videoTimeResult.isInGap);
  }

  // Update current frame when video time changes and handle windowed loading
  useEffect(() => {
    if (!videoElement || !currentVideo) return;

    const updateFrame = () => {
      const fps = currentVideo.fps || 30;
      const frame = Math.floor(videoElement.currentTime * fps);
      setCurrentFrame(frame);

      // Handle windowed detection loading
      if (!showDetections || !loadedRange) return;

      // Check if we seeked outside the loaded range
      if (frame < loadedRange.start || frame > loadedRange.end) {
        // Seek outside loaded range - fetch new window centered on current frame
        fetchWindow(currentVideo.id, frame, 'replace');
        return;
      }

      // Check if approaching end of loaded range (forward playback)
      if (frame > loadedRange.end - BUFFER_THRESHOLD) {
        fetchWindow(currentVideo.id, frame, 'extend-forward');
        return;
      }

      // Check if approaching start of loaded range (backward playback)
      if (frame < loadedRange.start + BUFFER_THRESHOLD && loadedRange.start > 0) {
        fetchWindow(currentVideo.id, frame, 'extend-backward');
      }
    };

    // Update on time update
    videoElement.addEventListener('timeupdate', updateFrame);
    updateFrame(); // Initial update

    return () => {
      videoElement.removeEventListener('timeupdate', updateFrame);
    };
  }, [videoElement, currentVideo, showDetections, loadedRange, fetchWindow]);

  const toggleDetections = useCallback(() => {
    const newValue = !showDetections;
    setShowDetections(newValue);

    if (newValue && currentVideo) {
      // Windowed detection loading - center on current frame
      fetchWindow(currentVideo.id, currentFrame, 'replace');

      // Jersey numbers are small, load all at once
      jerseyNumbersAPI.getByTrack(currentVideo.id)
        .then(response => {
          const map = new Map<number, AggregatedJerseyNumber>();
          for (const track of response.tracks) {
            map.set(track.tracking_id, track);
          }
          setJerseyNumbers(map);
        })
        .catch(err => {
          console.error('Failed to load jersey numbers:', err);
          setJerseyNumbers(new Map());
        });
    } else if (!newValue) {
      setDetections([]);
      setLoadedRange(null);
      setJerseyNumbers(new Map());
    }
  }, [showDetections, currentVideo, currentFrame, fetchWindow]);

  // Calculate total duration
  const totalDuration = useMemo(
    () =>
      videos.reduce((max, video) => {
        const end = (video.game_time_offset ?? 0) + video.duration_seconds;
        return Math.max(max, end);
      }, 0),
    [videos]
  );

  // Format time as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Frame-by-frame navigation (assuming 30 FPS)
  const FRAME_DURATION = 1 / 30;

  const stepForward = () => {
    seekToGameTime(Math.min(totalDuration, currentGameTime + FRAME_DURATION));
  };

  const stepBackward = () => {
    seekToGameTime(Math.max(0, currentGameTime - FRAME_DURATION));
  };

  const skipForward = (seconds: number = 10) => {
    seekToGameTime(Math.min(totalDuration, currentGameTime + seconds));
  };

  const skipBackward = (seconds: number = 10) => {
    seekToGameTime(Math.max(0, currentGameTime - seconds));
  };

  const togglePlayPause = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  const playbackSpeeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onPlayPause: togglePlayPause,
    onPreviousFrame: showAdvancedControls ? stepBackward : undefined,
    onNextFrame: showAdvancedControls ? stepForward : undefined,
    onSkipBackward: () => skipBackward(10),
    onSkipForward: () => skipForward(10),
    onIncreaseSpeed: () => {
      const currentIndex = playbackSpeeds.indexOf(playbackRate);
      if (currentIndex === -1 || currentIndex === playbackSpeeds.length - 1) return;
      setPlaybackRate(playbackSpeeds[currentIndex + 1]);
    },
    onDecreaseSpeed: () => {
      const currentIndex = playbackSpeeds.indexOf(playbackRate);
      if (currentIndex <= 0) return;
      setPlaybackRate(playbackSpeeds[currentIndex - 1]);
    },
  });

  return (
    <Paper radius="md" withBorder p="md" className={className} bg="dark.8">
      <Stack gap="md">
        <Box style={{ width: '100%', aspectRatio: '16/9', position: 'relative', backgroundColor: 'black' }}>
          <MultiVideoPlayer
            onVideoChange={handleVideoChange}
            onVideoElementChange={setVideoElement}
            containerRefCallback={(ref) => { videoContainerRef.current = ref; }}
            showBuffering={true}
            showGapIndicator={true}
          />

          {videoElement && videoContainerRef.current && (
            <DetectionOverlay
              videoElement={videoElement}
              detections={detections}
              currentFrame={currentFrame}
              visible={showDetections}
              minConfidence={0.35}
              showConfidence={true}
              showTrackingId={true}
              containerRef={videoContainerRef}
              jerseyNumbers={jerseyNumbers}
            />
          )}
        </Box>

        {/* Current video progress bar with seeking */}
        <VideoProgressBar
          currentVideo={currentVideo}
          isInGap={isInGap}
          height={8}
        />

        <Group justify="space-between" align="center">
          <Group gap="xs">
            {/* Detection toggle */}
            <Tooltip label={showDetections ? 'Hide detections' : 'Show detections'} withArrow>
              <Switch
                checked={showDetections}
                onChange={toggleDetections}
                size="md"
                onLabel={<IconEye size={16} />}
                offLabel={<IconEyeOff size={16} />}
                color="blue"
              />
            </Tooltip>

            <Tooltip label="Skip back 10s" withArrow>
              <ActionIcon
                variant="light"
                color="gray"
                size="lg"
                radius="xl"
                aria-label="Skip backward"
                onClick={() => skipBackward(10)}
              >
                <IconPlayerTrackPrev size={20} />
              </ActionIcon>
            </Tooltip>

            {showAdvancedControls && (
              <Tooltip label="Previous frame" withArrow>
                <ActionIcon
                  variant="light"
                  color="gray"
                  size="lg"
                  radius="xl"
                  aria-label="Previous frame"
                  onClick={stepBackward}
                >
                  <IconPlayerSkipBack size={20} />
                </ActionIcon>
              </Tooltip>
            )}
          </Group>

          <Group gap="sm">
            <Text fw={600} c="dimmed" ff="monospace">
              {formatTime(currentGameTime)} / {formatTime(totalDuration)}
            </Text>

            <Tooltip label={isPlaying ? 'Pause' : 'Play'} withArrow>
              <ActionIcon
                size="xl"
                radius="xl"
                color="blue"
                variant="filled"
                onClick={togglePlayPause}
                aria-label={isPlaying ? 'Pause playback' : 'Play playback'}
              >
                {isPlaying ? <IconPlayerPause size={26} /> : <IconPlayerPlay size={26} />}
              </ActionIcon>
            </Tooltip>
          </Group>

          <Group gap="xs">
            {showAdvancedControls && (
              <Tooltip label="Next frame" withArrow>
                <ActionIcon
                  variant="light"
                  color="gray"
                  size="lg"
                  radius="xl"
                  aria-label="Next frame"
                  onClick={stepForward}
                >
                  <IconPlayerSkipForward size={20} />
                </ActionIcon>
              </Tooltip>
            )}

            <Tooltip label="Skip forward 10s" withArrow>
              <ActionIcon
                variant="light"
                color="gray"
                size="lg"
                radius="xl"
                aria-label="Skip forward"
                onClick={() => skipForward(10)}
              >
                <IconPlayerTrackNext size={20} />
              </ActionIcon>
            </Tooltip>

            <Menu withinPortal position="top-end" shadow="md">
              <Menu.Target>
                <Button
                  variant="light"
                  color="gray"
                  size="xs"
                  leftSection={<IconGauge size={16} />}
                >
                  {playbackRate}x
                </Button>
              </Menu.Target>
              <Menu.Dropdown>
                {playbackSpeeds.map(rate => (
                  <Menu.Item
                    key={rate}
                    onClick={() => setPlaybackRate(rate)}
                    rightSection={rate === playbackRate ? <Text fw={600}>✔</Text> : undefined}
                  >
                    {rate}x
                  </Menu.Item>
                ))}
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>

        {showAdvancedControls && (
          <Text size="xs" c="dimmed" ta="center">
            Shortcuts: <Text span fw={600} ff="monospace">Space</Text> play/pause ·
            <Text span fw={600} ff="monospace">
              ←/→
            </Text>{' '}
            frame · <Text span fw={600} ff="monospace">J/L</Text> skip 10s ·{' '}
            <Text span fw={600} ff="monospace">↑/↓</Text> speed
          </Text>
        )}
      </Stack>
    </Paper>
  );
};

export default GameTimelinePlayer;
