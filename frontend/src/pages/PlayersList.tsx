/**
 * Players List Page
 *
 * Manage players database with CRUD operations
 */

import { useState, useEffect } from 'react';
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
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-xl">Loading players...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Navigation />
      <div className="p-8">
        {/* Header */}
        <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold">Players</h1>
          <button
            onClick={() => setShowCreateForm(true)}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-semibold transition-colors"
          >
            + Add Player
          </button>
        </div>

        {/* Search and filters */}
        <div className="mb-6 flex gap-4">
          <input
            type="text"
            placeholder="Search players..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <select
            value={teamFilter}
            onChange={(e) => setTeamFilter(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Teams</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
            {error}
          </div>
        )}

        {/* Players count */}
        <div className="mb-4 text-gray-400">
          Showing {filteredPlayers.length} of {players.length} players
        </div>

        {/* Players table */}
        {filteredPlayers.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            {searchTerm || teamFilter
              ? 'No players match your search criteria'
              : 'No players yet. Add your first player to get started.'}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold">Name</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">Jersey #</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">Team</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">Notes</th>
                  <th className="px-6 py-3 text-right text-sm font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {filteredPlayers.map((player) => (
                  <tr key={player.id} className="hover:bg-gray-700/50 transition-colors">
                    <td className="px-6 py-4 font-medium">{player.name}</td>
                    <td className="px-6 py-4">#{player.jersey_number}</td>
                    <td className="px-6 py-4">{player.team}</td>
                    <td className="px-6 py-4 text-gray-400 max-w-md truncate">
                      {player.notes || '-'}
                    </td>
                    <td className="px-6 py-4 text-right space-x-2">
                      <button
                        onClick={() => openEditModal(player)}
                        className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded transition-colors"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeletePlayer(player.id, player.name)}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded transition-colors"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create/Edit Modal */}
      {(showCreateForm || editingPlayer) && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              closeModal();
            }
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="player-form-title"
            className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md"
          >
            <h2 id="player-form-title" className="text-2xl font-bold mb-4">
              {editingPlayer ? 'Edit Player' : 'Add New Player'}
            </h2>

            <form onSubmit={editingPlayer ? handleEditPlayer : handleCreatePlayer}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1" htmlFor="player-name">
                    Player Name *
                  </label>
                  <input
                    id="player-name"
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="John Doe"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1" htmlFor="jersey-number">
                    Jersey Number *
                  </label>
                  <input
                    id="jersey-number"
                    type="number"
                    required
                    min="0"
                    max="99"
                    value={formData.jersey_number}
                    onChange={(e) =>
                      setFormData({ ...formData, jersey_number: parseInt(e.target.value) || 0 })
                    }
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="23"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1" htmlFor="team">
                    Team *
                  </label>
                  <input
                    id="team"
                    type="text"
                    required
                    value={formData.team}
                    onChange={(e) => setFormData({ ...formData, team: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Warriors"
                    list="team-suggestions"
                  />
                  <datalist id="team-suggestions">
                    {teams.map((team) => (
                      <option key={team} value={team} />
                    ))}
                  </datalist>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1" htmlFor="notes">
                    Notes
                  </label>
                  <textarea
                    id="notes"
                    value={formData.notes}
                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Optional notes about the player"
                    rows={3}
                  />
                </div>
              </div>

              <div className="mt-6 flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={closeModal}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded transition-colors"
                  disabled={isCreating}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreating}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCreating
                    ? editingPlayer
                      ? 'Updating...'
                      : 'Creating...'
                    : editingPlayer
                    ? 'Update Player'
                    : 'Create Player'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
