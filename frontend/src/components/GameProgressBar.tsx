/**
 * Game Progress Bar Component
 *
 * Interactive progress bar for the full game timeline with:
 * - Video segment visualization (color-coded)
 * - Gap visualization
 * - Annotation markers
 * - Click/drag to seek to any position
 * - Current position indicator
 */

import { useRef, useCallback, useState, useMemo } from 'react';
import { Box, Text, Tooltip } from '@mantine/core';
import { useTimelineStore, useVideoSegments, useVideoGaps } from '../store/timelineStore';
import type { Annotation } from '../types/api';

interface GameProgressBarProps {
  /** Annotations to display as markers */
  annotations: Annotation[];
  
  /** Height of the progress bar */
  height?: number;
  
  /** Show annotation markers */
  showAnnotations?: boolean;
  
  /** Callback when an annotation marker is clicked */
  onAnnotationClick?: (annotation: Annotation) => void;
  
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
 * Game Progress Bar - Full game timeline with video segments and annotation markers
 */
export const GameProgressBar: React.FC<GameProgressBarProps> = ({
  annotations,
  height = 24,
  showAnnotations = true,
  onAnnotationClick,
  className = '',
}) => {
  const barRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState<number>(0);

  // Timeline store
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const videos = useTimelineStore(state => state.videos);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const videoSegments = useVideoSegments();
  const videoGaps = useVideoGaps();

  // Calculate total duration
  const totalDuration = useMemo(
    () =>
      videos.reduce((max, video) => {
        const end = (video.game_time_offset ?? 0) + video.duration_seconds;
        return Math.max(max, end);
      }, 0),
    [videos]
  );

  // Calculate progress percentage
  const progressPercent = totalDuration > 0 ? (currentGameTime / totalDuration) * 100 : 0;

  // Colors for video segments
  const segmentColors = ['#228be6', '#40c057', '#be4bdb', '#f06595', '#4c6ef5', '#20c997'];

  // Calculate time from mouse position
  const getTimeFromPosition = useCallback(
    (clientX: number): number => {
      if (!barRef.current || totalDuration <= 0) return 0;
      const rect = barRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percent = x / rect.width;
      return percent * totalDuration;
    },
    [totalDuration]
  );

  // Handle mouse down to start seeking
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      const time = getTimeFromPosition(e.clientX);
      seekToGameTime(time);
    },
    [getTimeFromPosition, seekToGameTime]
  );

  // Handle mouse move for hover preview and dragging
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!barRef.current) return;
      const rect = barRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setHoverPosition(x);
      setHoverTime(getTimeFromPosition(e.clientX));

      if (isDragging) {
        const time = getTimeFromPosition(e.clientX);
        seekToGameTime(time);
      }
    },
    [getTimeFromPosition, isDragging, seekToGameTime]
  );

  // Handle mouse up to stop dragging
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

  // Get annotation color based on type
  const getAnnotationColor = (type: string): string => {
    switch (type) {
      case 'play':
        return '#40c057'; // green
      case 'event':
        return '#fab005'; // yellow
      case 'note':
        return '#be4bdb'; // purple
      default:
        return '#868e96'; // gray
    }
  };

  return (
    <Box className={className}>
      {/* Progress bar container */}
      <Box
        ref={barRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{
          position: 'relative',
          height,
          borderRadius: 'var(--mantine-radius-md)',
          overflow: 'hidden',
          background: 'var(--mantine-color-dark-6)',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        {/* Video segments (colored backgrounds) */}
        {videoSegments.map((segment, index) => {
          const leftPercent = totalDuration > 0 ? (segment.start / totalDuration) * 100 : 0;
          const widthPercent = totalDuration > 0 ? ((segment.end - segment.start) / totalDuration) * 100 : 0;
          const color = segmentColors[index % segmentColors.length];

          return (
            <Tooltip
              key={segment.videoId}
              label={`Video ${index + 1}: ${formatTime(segment.start)} - ${formatTime(segment.end)}`}
              position="top"
              withArrow
            >
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                  background: color,
                  opacity: 0.35,
                  transition: 'opacity 150ms',
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLElement).style.opacity = '0.5';
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLElement).style.opacity = '0.35';
                }}
              />
            </Tooltip>
          );
        })}

        {/* Gaps in coverage */}
        {videoGaps.map((gap, index) => {
          const leftPercent = totalDuration > 0 ? (gap.start / totalDuration) * 100 : 0;
          const widthPercent = totalDuration > 0 ? ((gap.end - gap.start) / totalDuration) * 100 : 0;

          return (
            <Tooltip
              key={`gap-${index}`}
              label={`Gap: ${formatTime(gap.start)} - ${formatTime(gap.end)}`}
              position="top"
              withArrow
            >
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                  background: 'var(--mantine-color-dark-4)',
                  backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(0,0,0,0.1) 3px, rgba(0,0,0,0.1) 6px)',
                }}
              />
            </Tooltip>
          );
        })}

        {/* Annotation markers */}
        {showAnnotations &&
          annotations.map((annotation) => {
            const leftPercent = totalDuration > 0 ? (annotation.game_timestamp_start / totalDuration) * 100 : 0;
            const widthPercent = totalDuration > 0 
              ? ((annotation.game_timestamp_end - annotation.game_timestamp_start) / totalDuration) * 100 
              : 0;
            const color = getAnnotationColor(annotation.annotation_type);

            return (
              <Tooltip
                key={annotation.id}
                label={
                  <div>
                    <div style={{ fontWeight: 600 }}>{annotation.title || annotation.annotation_type}</div>
                    <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>
                      {formatTime(annotation.game_timestamp_start)} - {formatTime(annotation.game_timestamp_end)}
                    </div>
                    {annotation.description && (
                      <div style={{ fontSize: '0.75rem', opacity: 0.7, maxWidth: 200, marginTop: 4 }}>
                        {annotation.description}
                      </div>
                    )}
                  </div>
                }
                position="top"
                withArrow
                multiline
              >
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    onAnnotationClick?.(annotation);
                  }}
                  style={{
                    position: 'absolute',
                    height: '50%',
                    bottom: 2,
                    left: `${leftPercent}%`,
                    width: `max(${widthPercent}%, 6px)`,
                    background: color,
                    borderRadius: 'var(--mantine-radius-sm)',
                    opacity: annotation.verified ? 0.9 : 0.6,
                    cursor: 'pointer',
                    transition: 'transform 150ms, opacity 150ms',
                    zIndex: 10,
                  }}
                  onMouseEnter={(e) => {
                    (e.target as HTMLElement).style.transform = 'scaleY(1.2)';
                    (e.target as HTMLElement).style.opacity = '1';
                  }}
                  onMouseLeave={(e) => {
                    (e.target as HTMLElement).style.transform = 'scaleY(1)';
                    (e.target as HTMLElement).style.opacity = annotation.verified ? '0.9' : '0.6';
                  }}
                />
              </Tooltip>
            );
          })}

        {/* Progress indicator (played portion) */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: 0,
            width: `${progressPercent}%`,
            background: 'linear-gradient(90deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.25) 100%)',
            pointerEvents: 'none',
          }}
        />

        {/* Current position indicator (playhead) */}
        <div
          style={{
            position: 'absolute',
            top: -2,
            bottom: -2,
            left: `${progressPercent}%`,
            width: 3,
            background: 'white',
            borderRadius: 2,
            boxShadow: '0 0 4px rgba(0,0,0,0.5)',
            transform: 'translateX(-50%)',
            zIndex: 20,
            pointerEvents: 'none',
          }}
        />

        {/* Hover time preview */}
        {hoverTime !== null && !isDragging && (
          <div
            style={{
              position: 'absolute',
              top: -28,
              left: hoverPosition,
              transform: 'translateX(-50%)',
              background: 'var(--mantine-color-dark-7)',
              color: 'white',
              padding: '2px 8px',
              borderRadius: 'var(--mantine-radius-sm)',
              fontSize: '0.75rem',
              fontFamily: 'monospace',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 30,
            }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </Box>

      {/* Time labels */}
      <Box
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 4,
        }}
      >
        <Text size="xs" c="dimmed" ff="monospace">
          {formatTime(currentGameTime)}
        </Text>
        <Text size="xs" c="dimmed" ff="monospace">
          {formatTime(totalDuration)}
        </Text>
      </Box>
    </Box>
  );
};

export default GameProgressBar;
