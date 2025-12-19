import { useState, useEffect, useMemo, useCallback } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  Anchor,
  Badge,
  Box,
  Button,
  Center,
  Collapse,
  Container,
  Group,
  Loader,
  Paper,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconCalendar,
  IconChevronDown,
  IconChevronUp,
  IconMapPin,
  IconVideo,
} from '@tabler/icons-react';
import { Navigation } from '../components/Navigation';
import { GameTimelinePlayer } from '../components/GameTimelinePlayer';
import { GameProgressBar } from '../components/GameProgressBar';
import { AnnotationAccordion } from '../components/AnnotationAccordion';
import { useTimelineStore } from '../store/timelineStore';
import { api } from '../services/api';
import type { Game, Video, Annotation, CreateAnnotationDTO } from '../types/api';

export const VideoAnalysis: React.FC = () => {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();
  const gameIdNum = gameId ? parseInt(gameId, 10) : 0;

  const [game, setGame] = useState<Game | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [annotationsLoading, setAnnotationsLoading] = useState(false);

  const currentGameTime = useTimelineStore(state => state.currentGameTime);
  const loadVideos = useTimelineStore(state => state.loadVideos);
  const loadAnnotations = useTimelineStore(state => state.loadAnnotations);
  const seekToGameTime = useTimelineStore(state => state.seekToGameTime);
  const reset = useTimelineStore(state => state.reset);

  useEffect(() => {
    if (!gameIdNum) {
      setError('Invalid game ID');
      setIsLoading(false);
      return;
    }

    void loadGameData();

    return () => {
      reset();
    };
  }, [gameIdNum]);

  const loadGameData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const gameData = await api.games.get(gameIdNum);
      setGame(gameData);

      const videosData = await api.videos.listByGame(gameIdNum);
      setVideos(videosData);
      loadVideos(videosData);

      // Load annotations
      const annotationsData = await api.annotations.listByGame(gameIdNum);
      setAnnotations(annotationsData);
      loadAnnotations(annotationsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load game data');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const totalDuration = useMemo(
    () =>
      videos.reduce((max, video) => {
        const end = (video.game_time_offset ?? 0) + video.duration_seconds;
        return Math.max(max, end);
      }, 0),
    [videos]
  );

  const progressPercent = totalDuration > 0 ? (currentGameTime / totalDuration) * 100 : 0;

  // Annotation CRUD handlers
  const handleCreateAnnotation = useCallback(async (data: CreateAnnotationDTO) => {
    setAnnotationsLoading(true);
    try {
      const newAnnotation = await api.annotations.create(data);
      const updatedAnnotations = [...annotations, newAnnotation];
      setAnnotations(updatedAnnotations);
      loadAnnotations(updatedAnnotations);
    } catch (err) {
      console.error('Failed to create annotation:', err);
      throw err;
    } finally {
      setAnnotationsLoading(false);
    }
  }, [annotations, loadAnnotations]);

  const handleUpdateAnnotation = useCallback(async (id: number, data: Partial<CreateAnnotationDTO>) => {
    setAnnotationsLoading(true);
    try {
      const updated = await api.annotations.update(id, data);
      const updatedAnnotations = annotations.map(a => (a.id === updated.id ? updated : a));
      setAnnotations(updatedAnnotations);
      loadAnnotations(updatedAnnotations);
    } catch (err) {
      console.error('Failed to update annotation:', err);
      throw err;
    } finally {
      setAnnotationsLoading(false);
    }
  }, [annotations, loadAnnotations]);

  const handleDeleteAnnotation = useCallback(async (id: number) => {
    setAnnotationsLoading(true);
    try {
      await api.annotations.delete(id);
      const updatedAnnotations = annotations.filter(a => a.id !== id);
      setAnnotations(updatedAnnotations);
      loadAnnotations(updatedAnnotations);
    } catch (err) {
      console.error('Failed to delete annotation:', err);
      throw err;
    } finally {
      setAnnotationsLoading(false);
    }
  }, [annotations, loadAnnotations]);

  const handleVerifyAnnotation = useCallback(async (id: number) => {
    setAnnotationsLoading(true);
    try {
      const verified = await api.annotations.verify(id);
      const updatedAnnotations = annotations.map(a => (a.id === verified.id ? verified : a));
      setAnnotations(updatedAnnotations);
      loadAnnotations(updatedAnnotations);
    } catch (err) {
      console.error('Failed to verify annotation:', err);
      throw err;
    } finally {
      setAnnotationsLoading(false);
    }
  }, [annotations, loadAnnotations]);

  // Handle clicking an annotation on the progress bar
  const handleAnnotationClick = useCallback((annotation: Annotation) => {
    seekToGameTime(annotation.game_timestamp_start);
    setShowAnnotations(true);
  }, [seekToGameTime]);

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
          <Button leftSection={<IconArrowLeft size={18} />} onClick={() => navigate(-1)}>
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
          <Button leftSection={<IconArrowLeft size={18} />} onClick={() => navigate(-1)}>
            Go Back
          </Button>
        </Stack>
      </Center>
    );
  }

  return (
    <div>
      <Navigation />
      <Container size="xl" py="xl">
        <Stack gap="lg">
          {/* Header */}
          <Stack gap="xs">
            <Anchor component={Link} to="/" size="sm" c="blue.4">
              <Group gap="xs">
                <IconArrowLeft size={14} />
                <span>Back to Games</span>
              </Group>
            </Anchor>
            <Group justify="space-between" align="flex-start">
              <div>
                <Title order={1}>{game.name}</Title>
                <Group gap="lg" c="dimmed" wrap="wrap" mt="xs">
                  <Group gap={6}>
                    <IconCalendar size={18} />
                    <Text size="sm">{formatDate(game.date)}</Text>
                  </Group>
                  <Text size="sm">•</Text>
                  <Text size="sm">
                    {game.home_team} vs {game.away_team}
                  </Text>
                  {game.location && (
                    <>
                      <Text size="sm">•</Text>
                      <Group gap={6}>
                        <IconMapPin size={18} />
                        <Text size="sm">{game.location}</Text>
                      </Group>
                    </>
                  )}
                </Group>
              </div>
              <Group gap="sm">
                <Badge color="blue" size="lg" leftSection={<IconVideo size={16} />}>
                  {videos.length} video{videos.length === 1 ? '' : 's'}
                </Badge>
                {currentVideo && (
                  <Badge variant="light" color="gray">
                    {currentVideo.resolution} • {currentVideo.fps} FPS
                  </Badge>
                )}
              </Group>
            </Group>
          </Stack>

          {/* Video Player */}
          <Paper withBorder radius="lg" shadow="sm" style={{ overflow: 'hidden' }}>
            <GameTimelinePlayer
              onVideoChange={setCurrentVideo}
              showAdvancedControls
            />
          </Paper>

          {/* Game Progress Bar with Annotations */}
          <Paper withBorder radius="md" p="md">
            <Group justify="space-between" align="center" mb="sm">
              <div>
                <Text fw={600}>Game Timeline</Text>
                <Text size="sm" c="dimmed">
                  {formatDuration(currentGameTime)} / {formatDuration(totalDuration)} ({Math.round(progressPercent)}%)
                </Text>
              </div>
              <Group gap="xs">
                <Badge variant="light" size="sm">
                  {videos.length} video{videos.length !== 1 ? 's' : ''}
                </Badge>
                <Badge variant="light" color="green" size="sm">
                  {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                </Badge>
              </Group>
            </Group>
            <GameProgressBar
              annotations={annotations}
              height={32}
              showAnnotations={true}
              onAnnotationClick={handleAnnotationClick}
            />
          </Paper>

          {/* Annotations Section */}
          <Box>
            <Button
              variant="subtle"
              color="gray"
              fullWidth
              onClick={() => setShowAnnotations(prev => !prev)}
              rightSection={showAnnotations ? <IconChevronUp size={16} /> : <IconChevronDown size={16} />}
              styles={{
                root: {
                  justifyContent: 'space-between',
                  paddingLeft: 'var(--mantine-spacing-md)',
                  paddingRight: 'var(--mantine-spacing-md)',
                },
                inner: {
                  justifyContent: 'space-between',
                  width: '100%',
                },
              }}
            >
              <Group gap="xs">
                <Text fw={600}>{showAnnotations ? 'Hide' : 'Show'} Annotations</Text>
                <Badge variant="light" size="sm">
                  {annotations.length}
                </Badge>
              </Group>
            </Button>
            <Collapse in={showAnnotations}>
              <Box mt="md">
                <AnnotationAccordion
                  gameId={gameIdNum}
                  annotations={annotations}
                  onCreate={handleCreateAnnotation}
                  onUpdate={handleUpdateAnnotation}
                  onDelete={handleDeleteAnnotation}
                  onVerify={handleVerifyAnnotation}
                  isLoading={annotationsLoading}
                />
              </Box>
            </Collapse>
          </Box>

          {/* Back button */}
          <Group justify="flex-start">
            <Button variant="subtle" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate(-1)}>
              Back to Games
            </Button>
          </Group>
        </Stack>
      </Container>
    </div>
  );
};

export default VideoAnalysis;
