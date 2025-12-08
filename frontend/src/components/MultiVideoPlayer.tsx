/**
 * Multi-video player component
 *
 * Displays the currently active video from a multi-video sequence,
 * with smooth transitions and gap handling.
 */

import { useEffect, useRef, useState } from 'react';
import { useMultiVideoPlayback } from '../hooks/useMultiVideoPlayback';
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

  const {
    currentVideoElement,
    isBuffering,
    isInGap,
  } = useMultiVideoPlayback({
    onVideoChange: (video) => {
      setCurrentVideo(video);
      onVideoChange?.(video);
    },
  });

  // Mount/unmount video element in the DOM
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !currentVideoElement) return;

    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Add current video element
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
    <div className={`relative w-full h-full bg-black ${className}`}>
      {/* Video container */}
      <div
        ref={containerRef}
        className="w-full h-full"
        style={{ aspectRatio: '16/9' }}
      />

      {/* Buffering indicator */}
      {showBuffering && isBuffering && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin" />
            <span className="text-white text-sm font-medium">Buffering...</span>
          </div>
        </div>
      )}

      {/* Gap indicator */}
      {showGapIndicator && isInGap && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <div className="text-center">
            <div className="text-white text-lg font-semibold mb-2">
              No Video Available
            </div>
            <div className="text-gray-400 text-sm">
              Gap in video coverage at this time
            </div>
          </div>
        </div>
      )}

      {/* Current video info (for debugging) */}
      {currentVideo && (
        <div className="absolute top-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
          Video {currentVideo.sequence_order ?? '?'}: {currentVideo.file_path.split('/').pop()}
        </div>
      )}
    </div>
  );
};

export default MultiVideoPlayer;
