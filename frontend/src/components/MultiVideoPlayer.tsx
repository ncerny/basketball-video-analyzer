/**
 * Multi-video player component
 *
 * Displays the currently active video from a multi-video sequence,
 * with smooth transitions and gap handling.
 */

import { useEffect, useRef, useState } from 'react';
import { useMultiVideoPlayback } from '../hooks/useMultiVideoPlayback';
import { useTimelineStore } from '../store/timelineStore';
import type { Video } from '../types/timeline';

interface MultiVideoPlayerProps {
  /** CSS class name for the container */
  className?: string;

  /** Callback when video changes */
  onVideoChange?: (video: Video | null) => void;

  /** Show buffering indicator */
  showBuffering?: boolean;

  /** Show gap indicator */
  showGapIndicator?: boolean;
}

/**
 * Multi-video player component that handles seamless playback across multiple videos
 */
export const MultiVideoPlayer: React.FC<MultiVideoPlayerProps> = ({
  className = '',
  onVideoChange,
  showBuffering = true,
  showGapIndicator = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);

  // Get annotations from store for context
  const annotations = useTimelineStore(state => state.annotations);

  const {
    currentVideoElement,
    isBuffering,
    isInGap,
    gapDisplayInfo,
  } = useMultiVideoPlayback({
    onVideoChange: (video) => {
      setCurrentVideo(video);
      onVideoChange?.(video);
    },
  });

  // Find any annotation that overlaps with the current gap
  // An annotation overlaps if it starts before the gap ends AND ends after the gap starts
  const gapAnnotation = gapDisplayInfo ? annotations.find(a =>
    a.game_timestamp_start < gapDisplayInfo.gapEnd &&
    a.game_timestamp_end > gapDisplayInfo.gapStart
  ) : null;

  // Format seconds as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Mount/unmount video element in the DOM
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !currentVideoElement) return;

    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Add current video element with proper positioning
    currentVideoElement.style.position = 'absolute';
    currentVideoElement.style.top = '0';
    currentVideoElement.style.left = '0';
    currentVideoElement.style.width = '100%';
    currentVideoElement.style.height = '100%';
    currentVideoElement.style.objectFit = 'contain';
    container.appendChild(currentVideoElement);

    return () => {
      if (container.contains(currentVideoElement)) {
        container.removeChild(currentVideoElement);
      }
    };
  }, [currentVideoElement]);

  return (
    <div 
      className={`${className}`} 
      style={{ 
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'black',
      }}
    >
      {/* Video container - hidden during gaps */}
      <div
        ref={containerRef}
        style={{ 
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: isInGap && gapDisplayInfo ? 'none' : 'block',
        }}
      />

      {/* Buffering indicator */}
      {showBuffering && isBuffering && !isInGap && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin" />
            <span className="text-white text-sm font-medium">Buffering...</span>
          </div>
        </div>
      )}

      {/* Gap indicator with countdown - covers entire video area */}
      {showGapIndicator && isInGap && gapDisplayInfo && (
        <div 
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#4a5568', // gray-600
          }}
        >
          <div style={{ textAlign: 'center', maxWidth: '32rem', padding: '0 2rem' }}>
            {/* Main title */}
            <div style={{ color: '#a0aec0', fontSize: '1.125rem', fontWeight: 500, marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Recording Gap
            </div>

            {/* Reason/annotation if available */}
            {gapAnnotation && (gapAnnotation.title || gapAnnotation.description) && (
              <div style={{ color: '#ecc94b', fontSize: '1.25rem', fontWeight: 500, marginBottom: '1.5rem', padding: '0.75rem 1rem', backgroundColor: 'rgba(116, 66, 16, 0.3)', borderRadius: '0.5rem', border: '1px solid rgba(214, 158, 46, 0.5)' }}>
                {gapAnnotation.title && (
                  <div style={{ marginBottom: gapAnnotation.description ? '0.5rem' : 0 }}>
                    {gapAnnotation.title}
                  </div>
                )}
                {gapAnnotation.description && (
                  <div style={{ fontSize: '1rem', opacity: 0.9 }}>
                    {gapAnnotation.description}
                  </div>
                )}
              </div>
            )}

            {/* Gap duration info */}
            <div style={{ color: '#718096', fontSize: '1rem', marginBottom: '1rem' }}>
              {gapDisplayInfo.actualGapDuration > gapDisplayInfo.displayDuration ? (
                <>Gap duration: {formatTime(gapDisplayInfo.actualGapDuration)} (fast-forwarding)</>
              ) : (
                <>Gap duration: {formatTime(gapDisplayInfo.actualGapDuration)}</>
              )}
            </div>

            {/* Countdown - the main focus */}
            <div style={{ color: 'white', fontSize: '2.25rem', fontWeight: 700, marginBottom: '1.5rem' }}>
              Resuming in {Math.ceil(gapDisplayInfo.displayRemaining)}s
            </div>

            {/* Progress bar */}
            <div style={{ width: '100%', backgroundColor: '#2d3748', borderRadius: '9999px', height: '0.5rem', overflow: 'hidden', marginBottom: '1rem' }}>
              <div 
                style={{ 
                  height: '100%', 
                  backgroundColor: '#4299e1',
                  transition: 'all 100ms',
                  width: `${(gapDisplayInfo.displayElapsed / gapDisplayInfo.displayDuration) * 100}%` 
                }}
              />
            </div>

            {/* Timeline position */}
            <div style={{ color: '#4a5568', fontSize: '0.875rem', fontFamily: 'monospace' }}>
              {formatTime(gapDisplayInfo.gapStart)} → {formatTime(gapDisplayInfo.gapEnd)}
            </div>
          </div>
        </div>
      )}

      {/* Current video info (for debugging) - hidden during gaps */}
      {currentVideo && !isInGap && (
        <div className="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
          Video {currentVideo.sequence_order ?? '?'}: {currentVideo.file_path.split('/').pop()}
        </div>
      )}
    </div>
  );
};

export default MultiVideoPlayer;
