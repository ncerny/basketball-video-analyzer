/**
 * Timeline Bar Component
 *
 * Visualizes the unified game timeline with:
 * - Video segments (color-coded)
 * - Gaps in coverage (gray)
 * - Overlapping videos (striped pattern)
 * - Annotation markers
 * - Interactive scrubbing
 * - Hover tooltips
 */

import { useMemo } from 'react';
import { useVideoSegments, useVideoGaps, useVideoOverlaps } from '../store/timelineStore';
import type { Annotation } from '../types/timeline';

export interface TimelineBarProps {
  /** Current playback position in game time (seconds) */
  currentTime: number;

  /** Total duration of the game timeline (seconds) */
  totalDuration: number;

  /** Callback when user seeks to a new time */
  onSeek: (time: number) => void;

  /** Annotations to display on timeline */
  annotations?: Annotation[];

  /** Show annotation markers */
  showAnnotations?: boolean;

  /** Height of the timeline bar in pixels */
  height?: number;

  /** CSS class name for the container */
  className?: string;

  /** Show time labels */
  showTimeLabels?: boolean;

  /** Show segment statistics */
  showStats?: boolean;

  /** Enable interactive scrubbing */
  interactive?: boolean;
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
 * TimelineBar - Reusable timeline visualization component
 */
export const TimelineBar: React.FC<TimelineBarProps> = ({
  currentTime,
  totalDuration,
  onSeek,
  annotations = [],
  showAnnotations = true,
  height = 48,
  className = '',
  showTimeLabels = true,
  showStats = true,
  interactive = true,
}) => {
  // Get video segments, gaps, and overlaps from store
  const videoSegments = useVideoSegments();
  const videoGaps = useVideoGaps();
  const videoOverlaps = useVideoOverlaps();

  // Colors for video segments (cycle through for multiple videos)
  const segmentColors = [
    'bg-blue-500',
    'bg-green-500',
    'bg-purple-500',
    'bg-pink-500',
    'bg-indigo-500',
    'bg-teal-500',
  ];

  // Handle click/drag on timeline
  const handleTimelineInteraction = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!interactive) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newTime = percent * totalDuration;
    onSeek(Math.max(0, Math.min(newTime, totalDuration)));
  };

  // Calculate current position percentage
  const currentPercent = totalDuration > 0 ? (currentTime / totalDuration) * 100 : 0;

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Time Display */}
      {showTimeLabels && (
        <div className="flex justify-between text-sm text-gray-300 px-1">
          <span className="font-mono">{formatTime(currentTime)}</span>
          <span className="font-mono text-gray-500">{formatTime(totalDuration)}</span>
        </div>
      )}

      {/* Timeline Track */}
      <div className="relative">
        {/* Background track */}
        <div
          className={`bg-gray-700 rounded-lg relative overflow-hidden ${interactive ? 'cursor-pointer' : ''}`}
          style={{ height: `${height}px` }}
          onClick={handleTimelineInteraction}
          role="slider"
          aria-valuemin={0}
          aria-valuemax={totalDuration}
          aria-valuenow={currentTime}
          aria-label="Timeline scrubber"
          tabIndex={interactive ? 0 : -1}
        >
          {/* Video segments */}
          {videoSegments.map((segment, index) => {
            const leftPercent = totalDuration > 0 ? (segment.start / totalDuration) * 100 : 0;
            const widthPercent = totalDuration > 0 ? ((segment.end - segment.start) / totalDuration) * 100 : 0;
            const colorClass = segmentColors[index % segmentColors.length];

            return (
              <div
                key={segment.videoId}
                className={`absolute top-0 bottom-0 ${colorClass} opacity-40 hover:opacity-60 transition-opacity`}
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
                title={`Video ${index + 1}: ${formatTime(segment.start)} - ${formatTime(segment.end)} (${formatTime(segment.duration)})`}
              />
            );
          })}

          {/* Gaps in coverage */}
          {videoGaps.map((gap, index) => {
            const leftPercent = totalDuration > 0 ? (gap.start / totalDuration) * 100 : 0;
            const widthPercent = totalDuration > 0 ? ((gap.end - gap.start) / totalDuration) * 100 : 0;

            return (
              <div
                key={`gap-${index}`}
                className="absolute top-0 bottom-0 bg-gray-500 opacity-30"
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                }}
                title={`Gap: ${formatTime(gap.start)} - ${formatTime(gap.end)}`}
              />
            );
          })}

          {/* Overlapping videos (striped pattern) */}
          {videoOverlaps.map((overlap, index) => {
            const leftPercent = totalDuration > 0 ? (overlap.start / totalDuration) * 100 : 0;
            const widthPercent = totalDuration > 0 ? ((overlap.end - overlap.start) / totalDuration) * 100 : 0;

            return (
              <div
                key={`overlap-${index}`}
                className="absolute top-0 bottom-0 bg-yellow-500 opacity-50"
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                  backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(0,0,0,0.1) 4px, rgba(0,0,0,0.1) 8px)',
                }}
                title={`Overlap: ${formatTime(overlap.start)} - ${formatTime(overlap.end)} (${overlap.videoIds.length} videos)`}
              />
            );
          })}

          {/* Annotation markers */}
          {showAnnotations && annotations.map((annotation) => {
            const startPercent = totalDuration > 0 ? (annotation.game_timestamp_start / totalDuration) * 100 : 0;
            const widthPercent = totalDuration > 0 ? ((annotation.game_timestamp_end - annotation.game_timestamp_start) / totalDuration) * 100 : 0;

            // Color based on annotation type
            const colorClass =
              annotation.annotation_type === 'play'
                ? 'bg-green-400'
                : annotation.annotation_type === 'event'
                ? 'bg-yellow-400'
                : 'bg-purple-400';

            // Opacity based on verification status
            const opacityClass = annotation.verified ? 'opacity-80' : 'opacity-50';

            return (
              <div
                key={annotation.id}
                className={`absolute top-1 h-2 ${colorClass} ${opacityClass} rounded-full cursor-pointer hover:opacity-100 transition-opacity`}
                style={{
                  left: `${startPercent}%`,
                  width: `${Math.max(widthPercent, 0.5)}%`,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  if (interactive) {
                    onSeek(annotation.game_timestamp_start);
                  }
                }}
                title={`${annotation.annotation_type} ${annotation.verified ? '✓' : '(unverified)'}: ${formatTime(annotation.game_timestamp_start)} - ${formatTime(annotation.game_timestamp_end)}`}
              />
            );
          })}

          {/* Invisible scrubber input for accessibility */}
          {interactive && (
            <input
              type="range"
              min="0"
              max={totalDuration}
              step="0.01"
              value={currentTime}
              onChange={(e) => onSeek(parseFloat(e.target.value))}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              aria-label="Seek timeline"
            />
          )}

          {/* Current time indicator */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-white shadow-lg pointer-events-none"
            style={{ left: `${currentPercent}%` }}
          >
            <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-3 h-3 bg-white rounded-full shadow-lg" />
          </div>
        </div>

        {/* Timeline statistics */}
        {showStats && (
          <div className="flex justify-between mt-1 px-1">
            <div className="text-xs text-gray-500">
              {videoSegments.length} video{videoSegments.length !== 1 ? 's' : ''}
              {videoGaps.length > 0 && `, ${videoGaps.length} gap${videoGaps.length !== 1 ? 's' : ''}`}
              {videoOverlaps.length > 0 && `, ${videoOverlaps.length} overlap${videoOverlaps.length !== 1 ? 's' : ''}`}
            </div>
            {showAnnotations && annotations.length > 0 && (
              <div className="text-xs text-gray-500">
                {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TimelineBar;
