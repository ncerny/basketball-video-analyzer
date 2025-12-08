/**
 * Demo component showing how to use the MultiVideoPlayer
 *
 * This demonstrates:
 * - Loading videos into the timeline
 * - Playback controls
 * - Seeking through the timeline
 * - Handling video transitions
 */

import { useEffect } from 'react';
import { useTimelineStore } from '../store/timelineStore';
import { MultiVideoPlayer } from './MultiVideoPlayer';
import type { Video } from '../types/timeline';

const GAME_ID = 1;

export const MultiVideoPlayerDemo: React.FC = () => {

  // Timeline store actions
  const loadVideos = useTimelineStore(state => state.loadVideos);
  const play = useTimelineStore(state => state.play);
  const pause = useTimelineStore(state => state.pause);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const setPlaybackRate = useTimelineStore(state => state.setPlaybackRate);

  // Timeline store state
  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const isPlaying = useTimelineStore(state => state.isPlaying);
  const playbackRate = useTimelineStore(state => state.playbackRate);
  const videos = useTimelineStore(state => state.videos);

  // Load videos for the game
  useEffect(() => {
    // In a real app, this would fetch from the API
    // For now, using mock data
    const mockVideos: Video[] = [
      {
        id: 1,
        game_id: GAME_ID,
        file_path: '/videos/game1_quarter1.mp4',
        duration_seconds: 600,
        fps: 30,
        resolution: '1920x1080',
        sequence_order: 0,
        game_time_offset: 0,
      },
      {
        id: 2,
        game_id: GAME_ID,
        file_path: '/videos/game1_quarter2.mp4',
        duration_seconds: 600,
        fps: 30,
        resolution: '1920x1080',
        sequence_order: 1,
        game_time_offset: 600,
      },
      {
        id: 3,
        game_id: GAME_ID,
        file_path: '/videos/game1_quarter3.mp4',
        duration_seconds: 600,
        fps: 30,
        resolution: '1920x1080',
        sequence_order: 2,
        game_time_offset: 1200,
      },
    ];

    loadVideos(mockVideos);
  }, [loadVideos]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const totalDuration = videos.reduce((max, video) => {
    const end = (video.game_time_offset ?? 0) + video.duration_seconds;
    return Math.max(max, end);
  }, 0);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow px-4 py-3">
        <h1 className="text-2xl font-bold text-gray-900">
          Multi-Video Playback Demo
        </h1>
        <p className="text-sm text-gray-600">
          Game #{GAME_ID} - {videos.length} video{videos.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Video player */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-4xl">
          <MultiVideoPlayer
            className="rounded-lg overflow-hidden shadow-lg"
            onVideoChange={(video) => {
              console.log('Video changed to:', video?.file_path);
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white border-t shadow-lg p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {/* Timeline scrubber */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-600">
              <span>{formatTime(currentGameTime)}</span>
              <span>{formatTime(totalDuration)}</span>
            </div>
            <input
              type="range"
              min="0"
              max={totalDuration}
              step="0.1"
              value={currentGameTime}
              onChange={(e) => seekToGameTime(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Playback controls */}
          <div className="flex items-center justify-center gap-4">
            {/* Skip back */}
            <button
              onClick={() => seekToGameTime(Math.max(0, currentGameTime - 10))}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg transition"
            >
              -10s
            </button>

            {/* Play/Pause */}
            <button
              onClick={() => isPlaying ? pause() : play()}
              className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>

            {/* Skip forward */}
            <button
              onClick={() => seekToGameTime(Math.min(totalDuration, currentGameTime + 10))}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg transition"
            >
              +10s
            </button>
          </div>

          {/* Playback rate */}
          <div className="flex items-center justify-center gap-2">
            <span className="text-sm text-gray-600">Speed:</span>
            {[0.5, 1.0, 1.5, 2.0].map((rate) => (
              <button
                key={rate}
                onClick={() => setPlaybackRate(rate)}
                className={`px-3 py-1 rounded transition ${
                  playbackRate === rate
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 hover:bg-gray-300'
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>

          {/* Video segments indicator */}
          <div className="space-y-1">
            <div className="text-xs text-gray-600 mb-1">Video Segments:</div>
            <div className="flex gap-1 h-2">
              {videos.map((video) => {
                const start = video.game_time_offset ?? 0;
                const end = start + video.duration_seconds;
                const leftPercent = (start / totalDuration) * 100;
                const widthPercent = ((end - start) / totalDuration) * 100;

                return (
                  <div
                    key={video.id}
                    className="bg-blue-400 rounded-sm relative"
                    style={{
                      marginLeft: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                    }}
                    title={`Video ${video.sequence_order}: ${formatTime(start)} - ${formatTime(end)}`}
                  />
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MultiVideoPlayerDemo;
