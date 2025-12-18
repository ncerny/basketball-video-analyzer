/**
 * Games List Page
 *
 * Displays all games with ability to:
 * - View list of all games
 * - Create new games
 * - Navigate to game analysis
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Title,
  Text,
  Button,
  Card,
  Group,
  Stack,
  Grid,
  Modal,
  TextInput,
  Loader,
  Center,
  Alert,
  Badge,
  ActionIcon,
} from '@mantine/core';
import {
  IconPlus,
  IconCalendar,
  IconUsers,
  IconMapPin,
  IconEye,
  IconVideo,
  IconAlertCircle,
  IconEdit,
} from '@tabler/icons-react';
import { Navigation } from '../components/Navigation';
import { api } from '../services/api';
import type { Game, CreateGameDTO } from '../types/api';

export const GamesList: React.FC = () => {
  const navigate = useNavigate();

  // State
  const [games, setGames] = useState<Game[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  // Form state
  const [formData, setFormData] = useState<CreateGameDTO>({
    name: '',
    date: new Date().toISOString().split('T')[0], // Today's date in YYYY-MM-DD format
    location: '',
    home_team: '',
    away_team: '',
  });

  // Load games on mount
  useEffect(() => {
    loadGames();
  }, []);

  // Load games from API
  const loadGames = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.games.list();
      // Sort by date descending (newest first)
      const sorted = data.sort((a, b) =>
        new Date(b.date).getTime() - new Date(a.date).getTime()
      );
      setGames(sorted);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load games');
      console.error('Error loading games:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Create new game
  const handleCreateGame = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    setError(null);

    try {
      const newGame = await api.games.create(formData);
      // Re-sort to maintain consistent date ordering
      setGames(prev => [newGame, ...prev].sort((a, b) =>
        new Date(b.date).getTime() - new Date(a.date).getTime()
      ));
      setShowCreateForm(false);
      setError(null); // Clear any previous errors on success
      // Reset form
      setFormData({
        name: '',
        date: new Date().toISOString().split('T')[0],
        location: '',
        home_team: '',
        away_team: '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create game');
      console.error('Error creating game:', err);
    } finally {
      setIsCreating(false);
    }
  };

  // Format date for display
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
      <Center h="100vh">
        <Stack align="center" gap="md">
          <Loader size="xl" />
          <Text size="xl" fw={600}>Loading games...</Text>
          <Text c="dimmed">Please wait</Text>
        </Stack>
      </Center>
    );
  }

  return (
    <div>
      <Navigation />

      {/* Header */}
      <Container size="xl" mb="xl">
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={1} size="h1">Basketball Video Analyzer</Title>
            <Text c="dimmed" size="lg" mt="xs">Manage and analyze your game footage</Text>
          </div>
          <Button
            leftSection={<IconPlus size={18} />}
            size="md"
            onClick={() => setShowCreateForm(true)}
          >
            New Game
          </Button>
        </Group>
      </Container>

      {/* Main Content */}
      <Container size="xl">
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

        {/* Create Game Modal */}
        <Modal
          opened={showCreateForm}
          onClose={() => setShowCreateForm(false)}
          title="Create New Game"
          size="md"
        >
          <form onSubmit={handleCreateGame}>
            <Stack gap="md">
              {/* Game Name */}
              <TextInput
                label="Game Name"
                placeholder="e.g., Warriors vs Lakers - Season Opener"
                required
                value={formData.name}
                onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
              />

              {/* Date */}
              <TextInput
                label="Date"
                type="date"
                required
                value={formData.date}
                onChange={e => setFormData(prev => ({ ...prev, date: e.target.value }))}
              />

              {/* Home Team */}
              <TextInput
                label="Home Team"
                placeholder="e.g., Golden State Warriors"
                required
                value={formData.home_team}
                onChange={e => setFormData(prev => ({ ...prev, home_team: e.target.value }))}
              />

              {/* Away Team */}
              <TextInput
                label="Away Team"
                placeholder="e.g., Los Angeles Lakers"
                required
                value={formData.away_team}
                onChange={e => setFormData(prev => ({ ...prev, away_team: e.target.value }))}
              />

              {/* Location */}
              <TextInput
                label="Location"
                placeholder="e.g., Chase Center, San Francisco"
                value={formData.location}
                onChange={e => setFormData(prev => ({ ...prev, location: e.target.value }))}
              />

              {/* Form Actions */}
              <Group justify="flex-end" mt="md">
                <Button
                  variant="subtle"
                  onClick={() => setShowCreateForm(false)}
                  disabled={isCreating}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  loading={isCreating}
                >
                  Create Game
                </Button>
              </Group>
            </Stack>
          </form>
        </Modal>

        {/* Games List */}
        {games.length === 0 ? (
          <Center py="xl">
            <Stack align="center" gap="md">
              <IconVideo size={48} stroke={1.5} color="var(--mantine-color-dimmed)" />
              <Title order={3} c="dimmed">No games yet</Title>
              <Text c="dimmed">Create your first game to get started</Text>
              <Button onClick={() => setShowCreateForm(true)}>
                Create Game
              </Button>
            </Stack>
          </Center>
        ) : (
          <Grid>
            {games.map(game => (
              <Grid.Col key={game.id} span={{ base: 12, sm: 6, lg: 4 }}>
                <Card shadow="sm" padding="lg" radius="md" withBorder>
                  <Stack gap="md">
                    <Group justify="space-between" align="flex-start">
                      <Title order={3} lineClamp={2} style={{ flex: 1 }}>{game.name}</Title>
                      <ActionIcon
                        variant="subtle"
                        color="gray"
                        onClick={() => navigate(`/games/${game.id}?edit=true`)}
                        title="Edit game"
                      >
                        <IconEdit size={18} />
                      </ActionIcon>
                    </Group>

                    <Stack gap="xs">
                      <Group gap="xs">
                        <IconCalendar size={16} stroke={1.5} />
                        <Text size="sm" c="dimmed">{formatDate(game.date)}</Text>
                      </Group>

                      <Group gap="xs">
                        <IconUsers size={16} stroke={1.5} />
                        <Text size="sm" c="dimmed">{game.home_team} vs {game.away_team}</Text>
                      </Group>

                      {game.location && (
                        <Group gap="xs">
                          <IconMapPin size={16} stroke={1.5} />
                          <Text size="sm" c="dimmed" lineClamp={1}>{game.location}</Text>
                        </Group>
                      )}
                    </Stack>

                    {/* Actions */}
                    <Group gap="xs" grow>
                      <Button
                        variant="light"
                        leftSection={<IconEye size={18} />}
                        onClick={() => navigate(`/games/${game.id}`)}
                      >
                        Details
                      </Button>
                      <Button
                        leftSection={<IconVideo size={18} />}
                        onClick={() => navigate(`/games/${game.id}/analysis`)}
                      >
                        Analyze
                      </Button>
                    </Group>
                  </Stack>
                </Card>
              </Grid.Col>
            ))}
          </Grid>
        )}
      </Container>
    </div>
  );
};

export default GamesList;
