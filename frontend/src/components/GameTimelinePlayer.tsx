/**
 * Game Timeline Player UI Component
 *
 * Comprehensive video player with:
 * - Multi-video playback
 * - Unified timeline visualization
 * - Full playback controls
 * - Frame-by-frame navigation
 * - Annotation markers
 * - Video segment indicators
 */

import { useState } from 'react';
import { useTimelineStore, useVideoSegments, useVideoGaps } from '../store/timelineStore';
import { MultiVideoPlayer } from './MultiVideoPlayer';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import type { Video } from '../types/timeline';

interface GameTimelinePlayerProps {
  /** CSS class name for the container */
  className?: string;

  /** Show annotation markers on timeline */
  showAnnotations?: boolean;

  /** Show advanced controls (frame-by-frame, etc.) */
  showAdvancedControls?: boolean;

  /** Callback when video changes */
  onVideoChange?: (video: Video | null) => void;
}

/**
 * Game Timeline Player - Full-featured video player with timeline
 */
export const GameTimelinePlayer: React.FC<GameTimelinePlayerProps> = ({
  className = '',
  showAnnotations = true,
  showAdvancedControls = true,
  onVideoChange,
}) => {
  // Timeline store state
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const playbackRate = useTimelineStore(state => state.playbackRate);
  const videos = useTimelineStore(state => state.videos);
  const annotations = useTimelineStore(state => state.annotations);

  // Timeline store actions
  const play = useTimelineStore(state => state.play);
  const pause = useTimelineStore(state => state.pause);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const setPlaybackRate = useTimelineStore(state => state.setPlaybackRate);

  // Video segments and gaps
  const videoSegments = useVideoSegments();
  const videoGaps = useVideoGaps();

  // UI state
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);

  // Calculate total duration
  const totalDuration = videos.reduce((max, video) => {
    const end = (video.game_time_offset ?? 0) + video.duration_seconds;
    return Math.max(max, end);
  }, 0);

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

  const cycleSpeed = (direction: 'up' | 'down') => {
    const speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
    const currentIndex = speeds.indexOf(playbackRate);
    if (currentIndex === -1) return;

    if (direction === 'up') {
      const nextIndex = Math.min(speeds.length - 1, currentIndex + 1);
      setPlaybackRate(speeds[nextIndex]);
    } else {
      const prevIndex = Math.max(0, currentIndex - 1);
      setPlaybackRate(speeds[prevIndex]);
    }
  };

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onPlayPause: togglePlayPause,
    onPreviousFrame: showAdvancedControls ? stepBackward : undefined,
    onNextFrame: showAdvancedControls ? stepForward : undefined,
    onSkipBackward: () => skipBackward(10),
    onSkipForward: () => skipForward(10),
    onIncreaseSpeed: () => cycleSpeed('up'),
    onDecreaseSpeed: () => cycleSpeed('down'),
  });

  return (
    <div className={`flex flex-col bg-gray-900 ${className}`}>
      {/* Video Player */}
      <div className="relative">
        <MultiVideoPlayer
          onVideoChange={onVideoChange}
          showBuffering={true}
          showGapIndicator={true}
        />
      </div>

      {/* Timeline and Controls */}
      <div className="bg-gray-800 p-4 space-y-3">
        {/* Timeline Scrubber */}
        <div className="space-y-2">
          {/* Time Display */}
          <div className="flex justify-between text-sm text-gray-300 px-1">
            <span className="font-mono">{formatTime(currentGameTime)}</span>
            <span className="font-mono text-gray-500">{formatTime(totalDuration)}</span>
          </div>

          {/* Timeline Track */}
          <div className="relative">
            {/* Background track */}
            <div className="h-12 bg-gray-700 rounded-lg relative overflow-hidden">
              {/* Video segments */}
              {videoSegments.map((segment, index) => {
                const leftPercent = (segment.start / totalDuration) * 100;
                const widthPercent = ((segment.end - segment.start) / totalDuration) * 100;

                return (
                  <div
                    key={segment.videoId}
                    className="absolute top-0 bottom-0 bg-blue-500 opacity-30"
                    style={{
                      left: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                    }}
                    title={`Video ${index + 1}: ${formatTime(segment.start)} - ${formatTime(segment.end)}`}
                  />
                );
              })}

              {/* Gaps */}
              {videoGaps.map((gap, index) => {
                const leftPercent = (gap.start / totalDuration) * 100;
                const widthPercent = ((gap.end - gap.start) / totalDuration) * 100;

                return (
                  <div
                    key={`gap-${index}`}
                    className="absolute top-0 bottom-0 bg-red-500 opacity-20"
                    style={{
                      left: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                    }}
                    title={`Gap: ${formatTime(gap.start)} - ${formatTime(gap.end)}`}
                  />
                );
              })}

              {/* Annotation markers */}
              {showAnnotations && annotations.map((annotation) => {
                const startPercent = (annotation.game_timestamp_start / totalDuration) * 100;
                const widthPercent = ((annotation.game_timestamp_end - annotation.game_timestamp_start) / totalDuration) * 100;

                const colorClass = annotation.annotation_type === 'play'
                  ? 'bg-green-400'
                  : annotation.annotation_type === 'event'
                  ? 'bg-yellow-400'
                  : 'bg-purple-400';

                return (
                  <div
                    key={annotation.id}
                    className={`absolute top-1 h-2 ${colorClass} rounded-full opacity-70 cursor-pointer hover:opacity-100`}
                    style={{
                      left: `${startPercent}%`,
                      width: `${Math.max(widthPercent, 0.5)}%`,
                    }}
                    onClick={() => seekToGameTime(annotation.game_timestamp_start)}
                    title={`${annotation.annotation_type}: ${annotation.verified ? '✓' : '?'}`}
                  />
                );
              })}

              {/* Scrubber input */}
              <input
                type="range"
                min="0"
                max={totalDuration}
                step="0.01"
                value={currentGameTime}
                onChange={(e) => seekToGameTime(parseFloat(e.target.value))}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />

              {/* Current time indicator */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-white shadow-lg"
                style={{ left: `${(currentGameTime / totalDuration) * 100}%` }}
              >
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-3 h-3 bg-white rounded-full shadow-lg" />
              </div>
            </div>

            {/* Timeline labels */}
            <div className="flex justify-between mt-1 px-1">
              <div className="text-xs text-gray-500">
                {videoSegments.length} video{videoSegments.length !== 1 ? 's' : ''}
                {videoGaps.length > 0 && `, ${videoGaps.length} gap${videoGaps.length !== 1 ? 's' : ''}`}
              </div>
              {showAnnotations && annotations.length > 0 && (
                <div className="text-xs text-gray-500">
                  {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center justify-between">
          {/* Left: Skip buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => skipBackward(10)}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition flex items-center gap-1"
              title="Skip back 10s"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.333 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
              </svg>
              <span className="text-xs">10s</span>
            </button>

            {showAdvancedControls && (
              <button
                onClick={stepBackward}
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition"
                title="Previous frame"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            )}
          </div>

          {/* Center: Play/Pause */}
          <button
            onClick={togglePlayPause}
            className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition flex items-center gap-2"
          >
            {isPlaying ? (
              <>
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                </svg>
                Pause
              </>
            ) : (
              <>
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
                Play
              </>
            )}
          </button>

          {/* Right: Skip and speed controls */}
          <div className="flex items-center gap-2">
            {showAdvancedControls && (
              <button
                onClick={stepForward}
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition"
                title="Next frame"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}

            <button
              onClick={() => skipForward(10)}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition flex items-center gap-1"
              title="Skip forward 10s"
            >
              <span className="text-xs">10s</span>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
              </svg>
            </button>

            {/* Playback speed */}
            <div className="relative">
              <button
                onClick={() => setShowSpeedMenu(!showSpeedMenu)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition min-w-[4rem] text-sm font-medium"
              >
                {playbackRate}x
              </button>
              {showSpeedMenu && (
                <div className="absolute bottom-full right-0 mb-2 bg-gray-700 rounded-lg shadow-lg overflow-hidden">
                  {[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                    <button
                      key={rate}
                      onClick={() => {
                        setPlaybackRate(rate);
                        setShowSpeedMenu(false);
                      }}
                      className={`block w-full px-6 py-2 text-left text-sm hover:bg-gray-600 transition ${
                        playbackRate === rate ? 'bg-blue-600 text-white' : 'text-gray-300'
                      }`}
                    >
                      {rate}x
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Keyboard shortcuts hint */}
        {showAdvancedControls && (
          <div className="text-xs text-gray-500 text-center pt-2 border-t border-gray-700">
            <span className="font-mono">Space</span>: Play/Pause
            <span className="mx-3">•</span>
            <span className="font-mono">←/→</span>: Frame
            <span className="mx-3">•</span>
            <span className="font-mono">J/L</span>: Skip 10s
            <span className="mx-3">•</span>
            <span className="font-mono">↑/↓</span>: Speed
          </div>
        )}
      </div>
    </div>
  );
};

export default GameTimelinePlayer;
