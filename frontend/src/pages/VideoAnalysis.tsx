/**
 * Video Analysis Page
 *
 * Complete video analysis interface with:
 * - Multi-video timeline player
 * - Annotation management panel
 * - Game/video information
 * - Resizable layout
 */

import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useTimelineStore } from '../store/timelineStore';
import { GameTimelinePlayer } from '../components/GameTimelinePlayer';
import { AnnotationPanel } from '../components/AnnotationPanel';
import { api } from '../services/api';
import type { Game, Video } from '../types/api';

/**
 * Video Analysis Page Component
 */
export const VideoAnalysis: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  const gameIdNum = gameId ? parseInt(gameId, 10) : 0;

  // State
  const [game, setGame] = useState<Game | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [annotationPanelWidth, setAnnotationPanelWidth] = useState(400);

  // Timeline store
  const loadVideos = useTimelineStore(state => state.loadVideos);
  const reset = useTimelineStore(state => state.reset);

  // Load game and videos on mount
  useEffect(() => {
    if (!gameIdNum) {
      setError('Invalid game ID');
      setIsLoading(false);
      return;
    }

    loadGameData();

    // Cleanup on unmount
    return () => {
      reset();
    };
  }, [gameIdNum]);

  // Load game and videos from API
  const loadGameData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load game info
      const gameData = await api.games.get(gameIdNum);
      setGame(gameData);

      // Load videos for this game
      const videosData = await api.videos.listByGame(gameIdNum);
      setVideos(videosData);

      // Load videos into timeline store
      loadVideos(videosData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load game data');
      console.error('Error loading game data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Format date
  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-xl font-semibold mb-2">Loading game...</div>
          <div className="text-gray-400">Please wait</div>
        </div>
      </div>
    );
  }

  if (error || !game) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-xl font-semibold mb-2 text-red-400">Error</div>
          <div className="text-gray-400">{error || 'Game not found'}</div>
          <button
            onClick={() => window.history.back()}
            className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (videos.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-xl font-semibold mb-2">No Videos</div>
          <div className="text-gray-400">
            No videos have been uploaded for this game yet
          </div>
          <button
            onClick={() => window.history.back()}
            className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{game.name}</h1>
            <div className="mt-1 flex items-center gap-4 text-sm text-gray-400">
              <span>{formatDate(game.date)}</span>
              <span>•</span>
              <span>
                {game.home_team} vs {game.away_team}
              </span>
              {game.location && (
                <>
                  <span>•</span>
                  <span>{game.location}</span>
                </>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowAnnotations(!showAnnotations)}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                showAnnotations
                  ? 'bg-blue-600 hover:bg-blue-700'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {showAnnotations ? 'Hide' : 'Show'} Annotations
            </button>
            <button
              onClick={() => window.history.back()}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
            >
              Back
            </button>
          </div>
        </div>

        {/* Video Info */}
        {currentVideo && (
          <div className="mt-3 p-3 bg-gray-900 rounded-lg text-sm">
            <div className="flex items-center gap-4">
              <span className="text-gray-400">Current Video:</span>
              <span className="font-medium">
                Video {videos.findIndex(v => v.id === currentVideo.id) + 1} of{' '}
                {videos.length}
              </span>
              <span className="text-gray-400">•</span>
              <span className="text-gray-400">
                {currentVideo.resolution} @ {currentVideo.fps} FPS
              </span>
              <span className="text-gray-400">•</span>
              <span className="text-gray-400">
                {Math.floor(currentVideo.duration_seconds / 60)}:
                {String(Math.floor(currentVideo.duration_seconds % 60)).padStart(2, '0')}
              </span>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Video Player (flexible) */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <GameTimelinePlayer
            onVideoChange={setCurrentVideo}
            showAnnotations={showAnnotations}
            showAdvancedControls={true}
            className="h-full"
          />
        </div>

        {/* Right: Annotation Panel (resizable) */}
        {showAnnotations && (
          <div
            className="border-l border-gray-700 flex-shrink-0 overflow-hidden"
            style={{ width: `${annotationPanelWidth}px` }}
          >
            <div className="h-full overflow-y-auto">
              <AnnotationPanel gameId={gameIdNum} />
            </div>

            {/* Resize Handle */}
            <div
              className="absolute top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500 transition-colors"
              style={{ left: `-1px` }}
              onMouseDown={e => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = annotationPanelWidth;

                const handleMouseMove = (e: MouseEvent) => {
                  const diff = startX - e.clientX;
                  const newWidth = Math.max(300, Math.min(800, startWidth + diff));
                  setAnnotationPanelWidth(newWidth);
                };

                const handleMouseUp = () => {
                  document.removeEventListener('mousemove', handleMouseMove);
                  document.removeEventListener('mouseup', handleMouseUp);
                };

                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoAnalysis;
