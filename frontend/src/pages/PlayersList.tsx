/**
 * Players List Page
 *
 * Manage players database with CRUD operations
 */

import { useState, useEffect } from 'react';
import {
  Container,
  Title,
  Text,
  Button,
  Group,
  Stack,
  TextInput,
  Select,
  Table,
  Modal,
  Alert,
  Center,
  Loader,
  Textarea,
  NumberInput,
  ActionIcon,
  Badge,
  Paper,
} from '@mantine/core';
import {
  IconPlus,
  IconSearch,
  IconEdit,
  IconTrash,
  IconAlertCircle,
  IconUsers,
} from '@tabler/icons-react';
import { Navigation } from '../components/Navigation';
import { api } from '../services/api';
import type { Player, CreatePlayerDTO } from '../types/api';

export function PlayersList() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [editingPlayer, setEditingPlayer] = useState<Player | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [teamFilter, setTeamFilter] = useState<string>('');

  // Form state
  const [formData, setFormData] = useState<CreatePlayerDTO>({
    name: '',
    jersey_number: 0,
    team: '',
    notes: '',
  });

  // Get unique teams for filter
  const teams = Array.from(new Set(players.map((p) => p.team))).sort();

  // Filter players
  const filteredPlayers = players.filter((player) => {
    const matchesSearch = player.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTeam = !teamFilter || player.team === teamFilter;
    return matchesSearch && matchesTeam;
  });

  // Fetch players
  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        setIsLoading(true);
        const data = await api.players.list();
        setPlayers(data.sort((a, b) => a.name.localeCompare(b.name)));
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load players');
      } finally {
        setIsLoading(false);
      }
    };

    fetchPlayers();
  }, []);

  // Escape key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && (showCreateForm || editingPlayer)) {
        setShowCreateForm(false);
        setEditingPlayer(null);
        resetForm();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [showCreateForm, editingPlayer]);

  const resetForm = () => {
    setFormData({
      name: '',
      jersey_number: 0,
      team: '',
      notes: '',
    });
  };

  const handleCreatePlayer = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    setError(null);

    try {
      const newPlayer = await api.players.create(formData);
      setPlayers((prev) =>
        [...prev, newPlayer].sort((a, b) => a.name.localeCompare(b.name))
      );
      setShowCreateForm(false);
      setError(null);
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create player');
    } finally {
      setIsCreating(false);
    }
  };

  const handleEditPlayer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingPlayer) return;

    setIsCreating(true);
    setError(null);

    try {
      const updated = await api.players.update(editingPlayer.id, formData);
      setPlayers((prev) =>
        prev
          .map((p) => (p.id === editingPlayer.id ? updated : p))
          .sort((a, b) => a.name.localeCompare(b.name))
      );
      setEditingPlayer(null);
      setError(null);
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update player');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeletePlayer = async (playerId: number, playerName: string) => {
    if (!confirm(`Are you sure you want to delete player "${playerName}"?`)) {
      return;
    }

    try {
      await api.players.delete(playerId);
      setPlayers((prev) => prev.filter((p) => p.id !== playerId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete player');
    }
  };

  const openEditModal = (player: Player) => {
    setEditingPlayer(player);
    setFormData({
      name: player.name,
      jersey_number: player.jersey_number,
      team: player.team,
      notes: player.notes || '',
    });
  };

  const closeModal = () => {
    setShowCreateForm(false);
    setEditingPlayer(null);
    resetForm();
  };

  if (isLoading) {
    return (
      <Center h="100vh">
        <Stack align="center" gap="md">
          <Loader size="xl" />
          <Text size="xl" fw={600}>Loading players...</Text>
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
            <Title order={1} size="h1">Players</Title>
            <Text c="dimmed" size="lg" mt="xs">Manage your player database</Text>
          </div>
          <Button
            leftSection={<IconPlus size={18} />}
            size="md"
            onClick={() => setShowCreateForm(true)}
          >
            Add Player
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

        {/* Search and Filters */}
        <Group mb="lg" grow>
          <TextInput
            placeholder="Search players..."
            leftSection={<IconSearch size={18} />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <Select
            placeholder="All Teams"
            leftSection={<IconUsers size={18} />}
            data={[
              { value: '', label: 'All Teams' },
              ...teams.map(team => ({ value: team, label: team }))
            ]}
            value={teamFilter}
            onChange={(value) => setTeamFilter(value || '')}
            clearable
          />
        </Group>

        {/* Players Count */}
        <Text c="dimmed" mb="md">
          Showing {filteredPlayers.length} of {players.length} players
        </Text>

        {/* Players Table */}
        {filteredPlayers.length === 0 ? (
          <Center py="xl">
            <Stack align="center" gap="md">
              <IconUsers size={48} stroke={1.5} color="var(--mantine-color-dimmed)" />
              <Title order={3} c="dimmed">
                {searchTerm || teamFilter
                  ? 'No players match your search criteria'
                  : 'No players yet'}
              </Title>
              {!searchTerm && !teamFilter && (
                <>
                  <Text c="dimmed">Add your first player to get started</Text>
                  <Button onClick={() => setShowCreateForm(true)}>
                    Add Player
                  </Button>
                </>
              )}
            </Stack>
          </Center>
        ) : (
          <Paper shadow="sm" withBorder>
            <Table.ScrollContainer minWidth={500}>
              <Table highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Jersey #</Table.Th>
                    <Table.Th>Team</Table.Th>
                    <Table.Th>Notes</Table.Th>
                    <Table.Th style={{ textAlign: 'right' }}>Actions</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {filteredPlayers.map((player) => (
                    <Table.Tr key={player.id}>
                      <Table.Td fw={500}>{player.name}</Table.Td>
                      <Table.Td>
                        <Badge variant="light" size="lg">#{player.jersey_number}</Badge>
                      </Table.Td>
                      <Table.Td>{player.team}</Table.Td>
                      <Table.Td c="dimmed" maw={300} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {player.notes || '-'}
                      </Table.Td>
                      <Table.Td>
                        <Group gap="xs" justify="flex-end">
                          <ActionIcon
                            variant="light"
                            color="blue"
                            onClick={() => openEditModal(player)}
                            title="Edit player"
                          >
                            <IconEdit size={18} />
                          </ActionIcon>
                          <ActionIcon
                            variant="light"
                            color="red"
                            onClick={() => handleDeletePlayer(player.id, player.name)}
                            title="Delete player"
                          >
                            <IconTrash size={18} />
                          </ActionIcon>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </Paper>
        )}

        {/* Create/Edit Modal */}
        <Modal
          opened={showCreateForm || editingPlayer !== null}
          onClose={closeModal}
          title={editingPlayer ? 'Edit Player' : 'Add New Player'}
          size="md"
        >
          <form onSubmit={editingPlayer ? handleEditPlayer : handleCreatePlayer}>
            <Stack gap="md">
              {/* Player Name */}
              <TextInput
                label="Player Name"
                placeholder="John Doe"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />

              {/* Jersey Number */}
              <NumberInput
                label="Jersey Number"
                placeholder="23"
                required
                min={0}
                max={99}
                value={formData.jersey_number}
                onChange={(value) => setFormData({ ...formData, jersey_number: Number(value) || 0 })}
              />

              {/* Team */}
              <TextInput
                label="Team"
                placeholder="Warriors"
                required
                value={formData.team}
                onChange={(e) => setFormData({ ...formData, team: e.target.value })}
                list="team-suggestions"
              />
              <datalist id="team-suggestions">
                {teams.map((team) => (
                  <option key={team} value={team} />
                ))}
              </datalist>

              {/* Notes */}
              <Textarea
                label="Notes"
                placeholder="Optional notes about the player"
                rows={3}
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              />

              {/* Form Actions */}
              <Group justify="flex-end" mt="md">
                <Button
                  variant="subtle"
                  onClick={closeModal}
                  disabled={isCreating}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  loading={isCreating}
                >
                  {editingPlayer ? 'Update Player' : 'Create Player'}
                </Button>
              </Group>
            </Stack>
          </form>
        </Modal>
      </Container>
    </div>
  );
}
