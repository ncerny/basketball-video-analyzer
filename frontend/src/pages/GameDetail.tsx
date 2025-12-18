/**
 * Game Detail Page
 *
 * Display game information, videos, and roster management
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom';
import {
  Container,
  Title,
  Text,
  Button,
  Group,
  Stack,
  Grid,
  Card,
  Paper,
  Modal,
  Alert,
  Center,
  Loader,
  Badge,
  FileButton,
  ActionIcon,
  Anchor,
  Radio,
  Divider,
  TextInput,
} from '@mantine/core';
import {
  IconUpload,
  IconPlus,
  IconTrash,
  IconAlertCircle,
  IconArrowLeft,
  IconCalendar,
  IconMapPin,
  IconUsers,
  IconVideo,
  IconClock,
  IconEdit,
  IconCheck,
  IconX,
  IconGripVertical,
} from '@tabler/icons-react';
import { Navigation } from '../components/Navigation';
import { VideoSequencer } from '../components/VideoSequencer';
import { api } from '../services/api';
import type { Game, Video, GameRoster, Player } from '../types/api';

interface RosterPlayerInfo extends GameRoster {
  player: Player;
}

export function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const [game, setGame] = useState<Game | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [roster, setRoster] = useState<RosterPlayerInfo[]>([]);
  const [allPlayers, setAllPlayers] = useState<Player[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showAddPlayerModal, setShowAddPlayerModal] = useState(false);
  const [selectedTeamSide, setSelectedTeamSide] = useState<'home' | 'away'>('home');
  const [isEditMode, setIsEditMode] = useState(false);

  // Edit mode state
  const [editedGame, setEditedGame] = useState<Partial<Game>>({});

  // Check for edit mode from URL parameter
  useEffect(() => {
    const editParam = searchParams.get('edit');
    if (editParam === 'true' && game) {
      setIsEditMode(true);
      setEditedGame({
        name: game.name,
        date: game.date,
        location: game.location,
        home_team: game.home_team,
        away_team: game.away_team,
      });
      // Remove the edit parameter from URL
      setSearchParams({});
    }
  }, [searchParams, game, setSearchParams]);

  useEffect(() => {
    if (!gameId) {
      navigate('/');
      return;
    }

    const fetchGameData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Fetch game details, videos, roster, and all players in parallel
        const [gameData, videosData, rosterData, playersData] = await Promise.all([
          api.games.get(parseInt(gameId)),
          api.videos.listByGame(parseInt(gameId)),
          api.gameRosters.listByGame(parseInt(gameId)),
          api.players.list(),
        ]);

        setGame(gameData);
        setVideos(videosData);
        setAllPlayers(playersData);

        // Enrich roster with player information
        const enrichedRoster: RosterPlayerInfo[] = rosterData.map((r: GameRoster) => {
          const player = playersData.find((p: Player) => p.id === r.player_id);
          return {
            ...r,
            player: player || {
              id: r.player_id,
              name: 'Unknown Player',
              jersey_number: 0,
              team: '',
              created_at: '',
            },
          };
        });

        setRoster(enrichedRoster);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load game data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchGameData();
  }, [gameId, navigate]);

  const handleVideoUpload = async (file: File | null) => {
    if (!file || !gameId) return;

    setIsUploading(true);
    setError(null);

    try {
      const newVideo = await api.videos.upload(parseInt(gameId), file);
      setVideos((prev) => [...prev, newVideo].sort((a, b) => (a.sequence_order || 0) - (b.sequence_order || 0)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload video');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteVideo = async (videoId: number) => {
    if (!confirm('Are you sure you want to delete this video?')) return;

    try {
      await api.videos.delete(videoId);
      setVideos((prev) => prev.filter((v) => v.id !== videoId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete video');
    }
  };

  const handleAddPlayerToRoster = async (playerId: number, teamSide: 'home' | 'away') => {
    if (!gameId) return;

    try {
      const newRosterEntry = await api.gameRosters.create({
        game_id: parseInt(gameId),
        player_id: playerId,
        team_side: teamSide,
      });

      const player = allPlayers.find((p) => p.id === playerId);
      if (player) {
        setRoster((prev) => [
          ...prev,
          {
            ...newRosterEntry,
            player,
          },
        ]);
      }
      setShowAddPlayerModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add player to roster');
    }
  };

  const handleRemovePlayerFromRoster = async (rosterId: number) => {
    if (!confirm('Remove this player from the roster?')) return;

    try {
      await api.gameRosters.delete(rosterId);
      setRoster((prev) => prev.filter((r) => r.id !== rosterId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove player from roster');
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const homeRoster = roster.filter((r) => r.team_side === 'home');
  const awayRoster = roster.filter((r) => r.team_side === 'away');
  const availablePlayers = allPlayers.filter(
    (p) => !roster.some((r) => r.player_id === p.id)
  );

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

  if (!game) {
    return (
      <Center h="100vh">
        <Stack align="center" gap="md">
          <Title order={1}>Game Not Found</Title>
          <Anchor component={Link} to="/" size="lg">
            Return to Games
          </Anchor>
        </Stack>
      </Center>
    );
  }

  return (
    <div>
      <Navigation />

      <Container size="xl">
        {/* Header */}
        <Stack gap="md" mb="xl">
          <Group justify="space-between">
            <Anchor component={Link} to="/" size="sm" c="blue">
              <Group gap="xs">
                <IconArrowLeft size={16} />
                <span>Back to Games</span>
              </Group>
            </Anchor>

            {!isEditMode ? (
              <Button
                leftSection={<IconEdit size={18} />}
                variant="light"
                onClick={() => {
                  if (!game) return;
                  setIsEditMode(true);
                  setEditedGame({
                    name: game.name,
                    date: game.date,
                    location: game.location || '',
                    home_team: game.home_team,
                    away_team: game.away_team,
                  });
                }}
              >
                Edit
              </Button>
            ) : (
              <Group gap="xs">
                <Button
                  leftSection={<IconCheck size={18} />}
                  color="green"
                  onClick={async () => {
                    if (!gameId) return;
                    try {
                      const updated = await api.games.update(parseInt(gameId), editedGame);
                      setGame(updated);
                      setIsEditMode(false);
                    } catch (err) {
                      setError(err instanceof Error ? err.message : 'Failed to save changes');
                    }
                  }}
                >
                  Save
                </Button>
                <Button
                  leftSection={<IconX size={18} />}
                  variant="light"
                  color="gray"
                  onClick={() => {
                    setIsEditMode(false);
                    setEditedGame({});
                  }}
                >
                  Cancel
                </Button>
              </Group>
            )}
          </Group>

          {!isEditMode ? (
            <Title order={1} size="h1">{game.name}</Title>
          ) : (
            <TextInput
              size="xl"
              value={editedGame.name || ''}
              onChange={(e) => setEditedGame(prev => ({ ...prev, name: e.target.value }))}
              styles={{
                input: {
                  fontSize: 'var(--mantine-h1-font-size)',
                  fontWeight: 700,
                  padding: '0.5rem',
                }
              }}
            />
          )}

          {!isEditMode ? (
            <Group gap="lg">
              <Group gap="xs">
                <IconCalendar size={18} />
                <Text c="dimmed">{formatDate(game.date)}</Text>
              </Group>
              {game.location && (
                <Group gap="xs">
                  <IconMapPin size={18} />
                  <Text c="dimmed">{game.location}</Text>
                </Group>
              )}
            </Group>
          ) : (
            <Group gap="md">
              <TextInput
                leftSection={<IconCalendar size={18} />}
                type="date"
                value={editedGame.date || ''}
                onChange={(e) => setEditedGame(prev => ({ ...prev, date: e.target.value }))}
                placeholder="Game date"
              />
              <TextInput
                leftSection={<IconMapPin size={18} />}
                value={editedGame.location || ''}
                onChange={(e) => setEditedGame(prev => ({ ...prev, location: e.target.value }))}
                placeholder="Location (optional)"
                style={{ flex: 1 }}
              />
            </Group>
          )}

          {!isEditMode ? (
            <Text size="lg" fw={500}>
              <Text span fw={700}>{game.home_team}</Text>
              <Text span c="dimmed"> vs </Text>
              <Text span fw={700}>{game.away_team}</Text>
            </Text>
          ) : (
            <Group gap="md" align="center">
              <TextInput
                value={editedGame.home_team || ''}
                onChange={(e) => setEditedGame(prev => ({ ...prev, home_team: e.target.value }))}
                placeholder="Home team"
                styles={{ input: { fontSize: 'var(--mantine-font-size-lg)', fontWeight: 700 } }}
              />
              <Text c="dimmed" fw={500}>vs</Text>
              <TextInput
                value={editedGame.away_team || ''}
                onChange={(e) => setEditedGame(prev => ({ ...prev, away_team: e.target.value }))}
                placeholder="Away team"
                styles={{ input: { fontSize: 'var(--mantine-font-size-lg)', fontWeight: 700 } }}
              />
            </Group>
          )}
        </Stack>

        {/* Error Message */}
        {error && (
          <Alert
            icon={<IconAlertCircle size={18} />}
            title="Error"
            color="red"
            withCloseButton
            onClose={() => setError(null)}
            mb="xl"
          >
            {error}
          </Alert>
        )}

        {/* Videos Section */}
        <Stack gap="md" mb="xl">
          <Group justify="space-between">
            <Group gap="xs">
              <IconVideo size={24} />
              <Title order={2}>Videos</Title>
              <Badge size="lg">{videos.length}</Badge>
            </Group>
            <FileButton
              onChange={handleVideoUpload}
              accept="video/*"
              disabled={isUploading}
            >
              {(props) => (
                <Button
                  {...props}
                  leftSection={<IconUpload size={18} />}
                  loading={isUploading}
                >
                  Upload Video
                </Button>
              )}
            </FileButton>
          </Group>

          {videos.length === 0 ? (
            <Center py="xl">
              <Stack align="center" gap="md">
                <IconVideo size={48} stroke={1.5} color="var(--mantine-color-dimmed)" />
                <Text c="dimmed">No videos uploaded yet</Text>
                <Text c="dimmed" size="sm">Upload a video to get started</Text>
              </Stack>
            </Center>
          ) : (
            <Grid>
              {videos.map((video) => (
                <Grid.Col key={video.id} span={{ base: 12, sm: 6, lg: 4 }}>
                  <Card shadow="sm" withBorder>
                    <Stack gap="sm">
                      <Group justify="space-between" align="flex-start">
                        <div>
                          <Text size="xs" c="dimmed">Video {video.sequence_order || video.id}</Text>
                          <Text size="xs" c="dimmed" lineClamp={1} style={{ fontFamily: 'monospace' }}>
                            {video.file_path.split('/').pop()}
                          </Text>
                        </div>
                        {isEditMode && (
                          <ActionIcon
                            color="red"
                            variant="subtle"
                            onClick={() => handleDeleteVideo(video.id)}
                            title="Delete video"
                          >
                            <IconTrash size={18} />
                          </ActionIcon>
                        )}
                      </Group>

                      <Divider />

                      <Stack gap="xs">
                        <Group justify="space-between">
                          <Group gap="xs">
                            <IconClock size={16} />
                            <Text size="sm" c="dimmed">Duration</Text>
                          </Group>
                          <Text size="sm">{formatDuration(video.duration_seconds)}</Text>
                        </Group>

                        <Group justify="space-between">
                          <Text size="sm" c="dimmed">Resolution</Text>
                          <Text size="sm">{video.resolution}</Text>
                        </Group>

                        <Group justify="space-between">
                          <Text size="sm" c="dimmed">Status</Text>
                          <Badge
                            color={
                              video.processing_status === 'completed'
                                ? 'green'
                                : video.processing_status === 'failed'
                                ? 'red'
                                : 'yellow'
                            }
                            variant="light"
                          >
                            {video.processing_status}
                          </Badge>
                        </Group>
                      </Stack>
                    </Stack>
                  </Card>
                </Grid.Col>
              ))}
            </Grid>
          )}

          {videos.length > 0 && isEditMode && (
            <VideoSequencer videos={videos} onVideosChange={setVideos} />
          )}
        </Stack>

        {/* Roster Section */}
        <Stack gap="md">
          <Group justify="space-between">
            <Group gap="xs">
              <IconUsers size={24} />
              <Title order={2}>Game Roster</Title>
            </Group>
            {isEditMode && (
              <Button
                leftSection={<IconPlus size={18} />}
                onClick={() => setShowAddPlayerModal(true)}
              >
                Add Player
              </Button>
            )}
          </Group>

          <Grid>
            {/* Home Team */}
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Paper shadow="sm" withBorder p="lg">
                <Group justify="space-between" mb="md">
                  <Title order={3}>{game.home_team}</Title>
                  <Badge size="lg">{homeRoster.length}</Badge>
                </Group>

                {homeRoster.length === 0 ? (
                  <Center py="md">
                    <Text c="dimmed">No players added</Text>
                  </Center>
                ) : (
                  <Stack gap="xs">
                    {homeRoster.map((r) => (
                      <Group
                        key={r.id}
                        justify="space-between"
                        p="sm"
                        style={{
                          backgroundColor: 'var(--mantine-color-dark-6)',
                          borderRadius: 'var(--mantine-radius-md)',
                        }}
                      >
                        <div>
                          <Text fw={500}>{r.player.name}</Text>
                          <Text size="sm" c="dimmed">
                            #{r.jersey_number_override || r.player.jersey_number}
                          </Text>
                        </div>
                        {isEditMode && (
                          <Button
                            variant="subtle"
                            color="red"
                            size="xs"
                            onClick={() => handleRemovePlayerFromRoster(r.id)}
                          >
                            Remove
                          </Button>
                        )}
                      </Group>
                    ))}
                  </Stack>
                )}
              </Paper>
            </Grid.Col>

            {/* Away Team */}
            <Grid.Col span={{ base: 12, lg: 6 }}>
              <Paper shadow="sm" withBorder p="lg">
                <Group justify="space-between" mb="md">
                  <Title order={3}>{game.away_team}</Title>
                  <Badge size="lg">{awayRoster.length}</Badge>
                </Group>

                {awayRoster.length === 0 ? (
                  <Center py="md">
                    <Text c="dimmed">No players added</Text>
                  </Center>
                ) : (
                  <Stack gap="xs">
                    {awayRoster.map((r) => (
                      <Group
                        key={r.id}
                        justify="space-between"
                        p="sm"
                        style={{
                          backgroundColor: 'var(--mantine-color-dark-6)',
                          borderRadius: 'var(--mantine-radius-md)',
                        }}
                      >
                        <div>
                          <Text fw={500}>{r.player.name}</Text>
                          <Text size="sm" c="dimmed">
                            #{r.jersey_number_override || r.player.jersey_number}
                          </Text>
                        </div>
                        {isEditMode && (
                          <Button
                            variant="subtle"
                            color="red"
                            size="xs"
                            onClick={() => handleRemovePlayerFromRoster(r.id)}
                          >
                            Remove
                          </Button>
                        )}
                      </Group>
                    ))}
                  </Stack>
                )}
              </Paper>
            </Grid.Col>
          </Grid>
        </Stack>

        {/* Add Player Modal */}
        <Modal
          opened={showAddPlayerModal}
          onClose={() => setShowAddPlayerModal(false)}
          title="Add Player to Roster"
          size="md"
        >
          <Stack gap="md">
            {/* Team Side Selection */}
            <div>
              <Text size="sm" fw={500} mb="xs">Team Side</Text>
              <Radio.Group
                value={selectedTeamSide}
                onChange={(value) => setSelectedTeamSide(value as 'home' | 'away')}
              >
                <Group>
                  <Radio value="home" label={game.home_team} />
                  <Radio value="away" label={game.away_team} />
                </Group>
              </Radio.Group>
            </div>

            <Divider />

            {/* Player Selection */}
            <div>
              <Text size="sm" fw={500} mb="xs">Select Player</Text>
              {availablePlayers.length === 0 ? (
                <Center py="md">
                  <Text c="dimmed">All players have been added to the roster</Text>
                </Center>
              ) : (
                <Stack gap="xs" mah={300} style={{ overflowY: 'auto' }}>
                  {availablePlayers.map((player) => (
                    <Button
                      key={player.id}
                      variant="light"
                      fullWidth
                      onClick={() => handleAddPlayerToRoster(player.id, selectedTeamSide)}
                      styles={{ label: { justifyContent: 'flex-start' } }}
                    >
                      <Stack gap={0}>
                        <Text fw={500}>{player.name}</Text>
                        <Text size="xs" c="dimmed">
                          #{player.jersey_number} • {player.team}
                        </Text>
                      </Stack>
                    </Button>
                  ))}
                </Stack>
              )}
            </div>

            <Group justify="flex-end">
              <Button variant="subtle" onClick={() => setShowAddPlayerModal(false)}>
                Close
              </Button>
            </Group>
          </Stack>
        </Modal>
      </Container>
    </div>
  );
}
