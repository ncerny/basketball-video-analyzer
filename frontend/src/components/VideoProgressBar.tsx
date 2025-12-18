/**
 * Video Progress Bar Component
 *
 * Shows progress within the current video or gap with:
 * - Video name display
 * - Progress within current video
 * - Click/drag to seek within video
 * - Gap indicator when in a gap
 */

import { useRef, useCallback, useState, useMemo } from 'react';
import { Box, Text, Tooltip } from '@mantine/core';
import { useTimelineStore } from '../store/timelineStore';
import type { Video } from '../types/timeline';

interface VideoProgressBarProps {
  /** Current video (null if in a gap) */
  currentVideo: Video | null;
  
  /** Whether currently in a gap */
  isInGap?: boolean;
  
  /** Height of the progress bar */
  height?: number;
  
  /** CSS class name */
  className?: string;
}

/**
 * Format seconds as MM:SS
 */
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Video Progress Bar - Shows progress within current video segment
 */
export const VideoProgressBar: React.FC<VideoProgressBarProps> = ({
  currentVideo,
  isInGap = false,
  height = 8,
  className = '',
}) => {
  const barRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState<number>(0);

  // Timeline store
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);

  // Calculate video progress
  const videoProgress = useMemo(() => {
    if (!currentVideo) return { videoTime: 0, percent: 0, duration: 0 };
    
    const offset = currentVideo.game_time_offset ?? 0;
    const videoTime = currentGameTime - offset;
    const duration = currentVideo.duration_seconds;
    const percent = duration > 0 ? (videoTime / duration) * 100 : 0;
    
    return {
      videoTime: Math.max(0, Math.min(videoTime, duration)),
      percent: Math.max(0, Math.min(percent, 100)),
      duration,
    };
  }, [currentVideo, currentGameTime]);

  // Calculate time from mouse position (relative to current video)
  const getVideoTimeFromPosition = useCallback(
    (clientX: number): number => {
      if (!barRef.current || !currentVideo) return 0;
      const rect = barRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percent = x / rect.width;
      return percent * currentVideo.duration_seconds;
    },
    [currentVideo]
  );

  // Handle mouse down to start seeking
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!currentVideo || isInGap) return;
      e.preventDefault();
      setIsDragging(true);
      
      const videoTime = getVideoTimeFromPosition(e.clientX);
      const offset = currentVideo.game_time_offset ?? 0;
      seekToGameTime(offset + videoTime);
    },
    [currentVideo, isInGap, getVideoTimeFromPosition, seekToGameTime]
  );

  // Handle mouse move
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!barRef.current || !currentVideo) return;
      
      const rect = barRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setHoverPosition(x);
      setHoverTime(getVideoTimeFromPosition(e.clientX));

      if (isDragging && !isInGap) {
        const videoTime = getVideoTimeFromPosition(e.clientX);
        const offset = currentVideo.game_time_offset ?? 0;
        seekToGameTime(offset + videoTime);
      }
    },
    [currentVideo, isInGap, getVideoTimeFromPosition, isDragging, seekToGameTime]
  );

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setHoverTime(null);
    if (isDragging) {
      setIsDragging(false);
    }
  }, [isDragging]);

  // If in a gap, show gap indicator
  if (isInGap || !currentVideo) {
    return (
      <Box className={className}>
        <Box
          style={{
            height,
            borderRadius: 'var(--mantine-radius-sm)',
            background: 'var(--mantine-color-dark-4)',
            backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(0,0,0,0.15) 3px, rgba(0,0,0,0.15) 6px)',
          }}
        />
        <Box
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 4,
          }}
        >
          <Text size="xs" c="dimmed">
            Gap in recording
          </Text>
          <Text size="xs" c="dimmed" ff="monospace">
            {formatTime(currentGameTime)}
          </Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box className={className}>
      {/* Progress bar */}
      <Tooltip
        label={`${formatTime(videoProgress.videoTime)} / ${formatTime(videoProgress.duration)}`}
        position="top"
        withArrow
        opened={isDragging}
      >
        <Box
          ref={barRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          style={{
            position: 'relative',
            height,
            borderRadius: 'var(--mantine-radius-sm)',
            background: 'var(--mantine-color-dark-5)',
            cursor: 'pointer',
            userSelect: 'none',
            overflow: 'hidden',
          }}
        >
          {/* Progress fill */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              bottom: 0,
              width: `${videoProgress.percent}%`,
              background: 'var(--mantine-color-blue-6)',
              borderRadius: 'var(--mantine-radius-sm)',
              transition: isDragging ? 'none' : 'width 100ms',
            }}
          />

          {/* Hover preview line */}
          {hoverTime !== null && !isDragging && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: hoverPosition,
                width: 2,
                background: 'rgba(255,255,255,0.5)',
                transform: 'translateX(-50%)',
                pointerEvents: 'none',
              }}
            />
          )}
        </Box>
      </Tooltip>

      {/* Time labels */}
      <Box
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: 4,
        }}
      >
        <Text size="xs" c="dimmed" ff="monospace">
          {formatTime(videoProgress.videoTime)}
        </Text>
        <Text size="xs" c="dimmed" ff="monospace">
          {formatTime(videoProgress.duration)}
        </Text>
      </Box>
    </Box>
  );
};

export default VideoProgressBar;
