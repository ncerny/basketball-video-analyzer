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
import {
  AppShell,
  Title,
  Text,
  Button,
  Group,
  Stack,
  Center,
  Loader,
  Paper,
  Badge,
  Box,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconEye,
  IconEyeOff,
  IconCalendar,
  IconMapPin,
  IconVideo,
} from '@tabler/icons-react';
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

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoading) {
    return (
      <Center h="100vh">
        <Stack align="center" gap="md">
          <Loader size="xl" />
          <Text size="xl" fw={600}>Loading game...</Text>
          <Text c="dimmed">Please wait</Text>
        </Stack>
      </Center>
    );
  }

  if (error || !game) {
    return (
      <Center h="100vh">
        <Stack align="center" gap="md">
          <Title order={2} c="red">Error</Title>
          <Text c="dimmed">{error || 'Game not found'}</Text>
          <Button
            leftSection={<IconArrowLeft size={18} />}
            onClick={() => window.history.back()}
          >
            Go Back
          </Button>
        </Stack>
      </Center>
    );
  }

  if (videos.length === 0) {
    return (
      <Center h="100vh">
        <Stack align="center" gap="md">
          <IconVideo size={48} stroke={1.5} color="var(--mantine-color-dimmed)" />
          <Title order={2}>No Videos</Title>
          <Text c="dimmed">No videos have been uploaded for this game yet</Text>
          <Button
            leftSection={<IconArrowLeft size={18} />}
            onClick={() => window.history.back()}
          >
            Go Back
          </Button>
        </Stack>
      </Center>
    );
  }

  return (
    <AppShell
      header={{ height: currentVideo ? 160 : 120 }}
      padding={0}
      styles={{
        main: {
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      <AppShell.Header
        style={{
          borderBottom: '1px solid var(--mantine-color-dark-4)',
        }}
      >
        <Box p="md">
          {/* Header */}
          <Group justify="space-between" mb="sm">
            <div>
              <Title order={2}>{game.name}</Title>
              <Group gap="xs" mt={4}>
                <Group gap={4}>
                  <IconCalendar size={16} />
                  <Text size="sm" c="dimmed">{formatDate(game.date)}</Text>
                </Group>
                <Text c="dimmed" size="sm">•</Text>
                <Text size="sm" c="dimmed">
                  {game.home_team} vs {game.away_team}
                </Text>
                {game.location && (
                  <>
                    <Text c="dimmed" size="sm">•</Text>
                    <Group gap={4}>
                      <IconMapPin size={16} />
                      <Text size="sm" c="dimmed">{game.location}</Text>
                    </Group>
                  </>
                )}
              </Group>
            </div>

            {/* Controls */}
            <Group gap="xs">
              <Button
                variant={showAnnotations ? 'filled' : 'light'}
                leftSection={showAnnotations ? <IconEyeOff size={18} /> : <IconEye size={18} />}
                onClick={() => setShowAnnotations(!showAnnotations)}
              >
                {showAnnotations ? 'Hide' : 'Show'} Annotations
              </Button>
              <Button
                variant="subtle"
                leftSection={<IconArrowLeft size={18} />}
                onClick={() => window.history.back()}
              >
                Back
              </Button>
            </Group>
          </Group>

          {/* Current Video Info */}
          {currentVideo && (
            <Paper p="xs" withBorder>
              <Group gap="md">
                <Group gap={4}>
                  <Text size="sm" c="dimmed">Current Video:</Text>
                  <Badge>
                    Video {videos.findIndex(v => v.id === currentVideo.id) + 1} of{' '}
                    {videos.length}
                  </Badge>
                </Group>
                <Text c="dimmed" size="sm">•</Text>
                <Text size="sm" c="dimmed">
                  {currentVideo.resolution} @ {currentVideo.fps} FPS
                </Text>
                <Text c="dimmed" size="sm">•</Text>
                <Text size="sm" c="dimmed">
                  {formatDuration(currentVideo.duration_seconds)}
                </Text>
              </Group>
            </Paper>
          )}
        </Box>
      </AppShell.Header>

      <AppShell.Main style={{ paddingTop: currentVideo ? '160px' : '120px' }}>
        {/* Main Content */}
        <Box
          style={{
            display: 'flex',
            height: '100%',
            overflow: 'hidden',
          }}
        >
          {/* Left: Video Player (flexible) */}
          <Box
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <GameTimelinePlayer
              onVideoChange={setCurrentVideo}
              showAnnotations={showAnnotations}
              showAdvancedControls={true}
              className="h-full"
            />
          </Box>

          {/* Right: Annotation Panel (resizable) */}
          {showAnnotations && (
            <Box
              style={{
                width: `${annotationPanelWidth}px`,
                borderLeft: '1px solid var(--mantine-color-dark-4)',
                flexShrink: 0,
                overflow: 'hidden',
                position: 'relative',
              }}
            >
              <Box
                style={{
                  height: '100%',
                  overflowY: 'auto',
                }}
              >
                <AnnotationPanel gameId={gameIdNum} />
              </Box>

              {/* Resize Handle */}
              <Box
                style={{
                  position: 'absolute',
                  top: 0,
                  bottom: 0,
                  left: -1,
                  width: 4,
                  cursor: 'col-resize',
                  transition: 'background-color 200ms',
                  ':hover': {
                    backgroundColor: 'var(--mantine-color-blue-6)',
                  },
                }}
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
            </Box>
          )}
        </Box>
      </AppShell.Main>
    </AppShell>
  );
};

export default VideoAnalysis;
