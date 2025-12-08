/**
 * Games List Page
 *
 * Displays all games with ability to:
 * - View list of all games
 * - Create new games
 * - Navigate to game analysis
 * - Delete games
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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

  // Handle Escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showCreateForm) {
        setShowCreateForm(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [showCreateForm]);

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

  // Delete game
  const handleDeleteGame = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to delete "${name}"?`)) {
      return;
    }

    try {
      await api.games.delete(id);
      setGames(prev => prev.filter(g => g.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete game');
      console.error('Error deleting game:', err);
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
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-xl font-semibold mb-2">Loading games...</div>
          <div className="text-gray-400">Please wait</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Navigation />
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Basketball Video Analyzer</h1>
              <p className="mt-1 text-gray-400">Manage and analyze your game footage</p>
            </div>
            <button
              onClick={() => setShowCreateForm(true)}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              New Game
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900 border border-red-700 rounded-lg text-red-200">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-sm underline hover:no-underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Create Game Form */}
        {showCreateForm && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={e => {
              // Close modal when clicking backdrop (not the content)
              if (e.target === e.currentTarget) {
                setShowCreateForm(false);
              }
            }}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-labelledby="create-game-title"
              className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md"
            >
              <h2 id="create-game-title" className="text-2xl font-bold mb-4">Create New Game</h2>

              <form onSubmit={handleCreateGame}>
                <div className="space-y-4">
                  {/* Game Name */}
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Game Name <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.name}
                      onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="e.g., Warriors vs Lakers - Season Opener"
                    />
                  </div>

                  {/* Date */}
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Date <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="date"
                      required
                      value={formData.date}
                      onChange={e => setFormData(prev => ({ ...prev, date: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  {/* Home Team */}
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Home Team <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.home_team}
                      onChange={e => setFormData(prev => ({ ...prev, home_team: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="e.g., Golden State Warriors"
                    />
                  </div>

                  {/* Away Team */}
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Away Team <span className="text-red-400">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.away_team}
                      onChange={e => setFormData(prev => ({ ...prev, away_team: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="e.g., Los Angeles Lakers"
                    />
                  </div>

                  {/* Location */}
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Location
                    </label>
                    <input
                      type="text"
                      value={formData.location}
                      onChange={e => setFormData(prev => ({ ...prev, location: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="e.g., Chase Center, San Francisco"
                    />
                  </div>
                </div>

                {/* Form Actions */}
                <div className="mt-6 flex gap-3">
                  <button
                    type="button"
                    onClick={() => setShowCreateForm(false)}
                    disabled={isCreating}
                    className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isCreating}
                    className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition disabled:opacity-50"
                  >
                    {isCreating ? 'Creating...' : 'Create Game'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Games List */}
        {games.length === 0 ? (
          <div className="text-center py-12">
            <svg
              className="mx-auto h-12 w-12 text-gray-600 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
            <h3 className="text-xl font-semibold text-gray-400 mb-2">No games yet</h3>
            <p className="text-gray-500 mb-4">Create your first game to get started</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
            >
              Create Game
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {games.map(game => (
              <div
                key={game.id}
                className="bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition overflow-hidden"
              >
                <div className="p-6">
                  <h3 className="text-xl font-semibold mb-2 line-clamp-2">{game.name}</h3>

                  <div className="space-y-2 text-sm text-gray-400 mb-4">
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                      <span>{formatDate(game.date)}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                        />
                      </svg>
                      <span>{game.home_team} vs {game.away_team}</span>
                    </div>

                    {game.location && (
                      <div className="flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                        </svg>
                        <span className="line-clamp-1">{game.location}</span>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => navigate(`/games/${game.id}/analysis`)}
                      className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition"
                    >
                      Analyze
                    </button>
                    <button
                      onClick={() => handleDeleteGame(game.id, game.name)}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition"
                      title="Delete game"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default GamesList;
