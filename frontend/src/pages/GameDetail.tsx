/**
 * Game Detail Page
 *
 * Display game information, videos, and roster management
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Navigation } from '../components/Navigation';
import { api } from '../services/api';
import type { Game, Video, GameRoster, Player } from '../types/api';

interface RosterPlayerInfo extends GameRoster {
  player: Player;
}

export function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();

  const [game, setGame] = useState<Game | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [roster, setRoster] = useState<RosterPlayerInfo[]>([]);
  const [allPlayers, setAllPlayers] = useState<Player[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showAddPlayerModal, setShowAddPlayerModal] = useState(false);
  const [selectedTeamSide, setSelectedTeamSide] = useState<'home' | 'away'>('home');

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
        setVideos((videosData as any).videos || []);
        setAllPlayers((playersData as any).players || []);

        // Enrich roster with player information
        const enrichedRoster: RosterPlayerInfo[] = ((rosterData as any).rosters || []).map((r: GameRoster) => {
          const player = ((playersData as any).players || []).find((p: Player) => p.id === r.player_id);
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

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0] || !gameId) return;

    const file = e.target.files[0];
    setIsUploading(true);
    setError(null);

    try {
      const newVideo = await api.videos.upload(parseInt(gameId), file);
      setVideos((prev) => [...prev, newVideo].sort((a, b) => (a.sequence_order || 0) - (b.sequence_order || 0)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload video');
    } finally {
      setIsUploading(false);
      // Reset file input
      e.target.value = '';
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
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-xl">Loading game...</div>
      </div>
    );
  }

  if (!game) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Game Not Found</h1>
          <Link to="/" className="text-blue-400 hover:text-blue-300">
            Return to Games
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Navigation />
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <Link to="/" className="text-blue-400 hover:text-blue-300 mb-4 inline-block">
              ← Back to Games
            </Link>
            <h1 className="text-4xl font-bold mb-2">{game.name}</h1>
            <div className="text-gray-400 space-y-1">
              <p>{formatDate(game.date)}</p>
              {game.location && <p>{game.location}</p>}
              <p className="text-lg">
                <span className="font-semibold">{game.home_team}</span> vs{' '}
                <span className="font-semibold">{game.away_team}</span>
              </p>
            </div>
          </div>

          {/* Error display */}
          {error && (
            <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
              {error}
            </div>
          )}

          {/* Videos Section */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Videos ({videos.length})</h2>
              <label className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-semibold transition-colors cursor-pointer">
                {isUploading ? 'Uploading...' : '+ Upload Video'}
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  disabled={isUploading}
                  className="hidden"
                />
              </label>
            </div>

            {videos.length === 0 ? (
              <div className="text-center py-12 text-gray-400 bg-gray-800 rounded-lg">
                No videos uploaded yet. Upload a video to get started.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {videos.map((video) => (
                  <div key={video.id} className="bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="text-sm text-gray-400">Video {video.sequence_order || video.id}</div>
                        <div className="text-sm font-mono text-gray-500 truncate">
                          {video.file_path.split('/').pop()}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteVideo(video.id)}
                        className="text-red-400 hover:text-red-300 ml-2"
                        title="Delete video"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Duration:</span>
                        <span>{formatDuration(video.duration_seconds)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Resolution:</span>
                        <span>{video.resolution}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Status:</span>
                        <span
                          className={
                            video.processing_status === 'completed'
                              ? 'text-green-400'
                              : video.processing_status === 'failed'
                              ? 'text-red-400'
                              : 'text-yellow-400'
                          }
                        >
                          {video.processing_status}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Roster Section */}
          <div>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold">Game Roster</h2>
              <button
                onClick={() => setShowAddPlayerModal(true)}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg font-semibold transition-colors"
              >
                + Add Player
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Home Team */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4">
                  {game.home_team} ({homeRoster.length})
                </h3>
                {homeRoster.length === 0 ? (
                  <div className="text-gray-400 text-center py-4">No players added</div>
                ) : (
                  <div className="space-y-2">
                    {homeRoster.map((r) => (
                      <div
                        key={r.id}
                        className="flex justify-between items-center bg-gray-700 rounded p-3"
                      >
                        <div>
                          <div className="font-semibold">{r.player.name}</div>
                          <div className="text-sm text-gray-400">
                            #{r.jersey_number_override || r.player.jersey_number}
                          </div>
                        </div>
                        <button
                          onClick={() => handleRemovePlayerFromRoster(r.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Away Team */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4">
                  {game.away_team} ({awayRoster.length})
                </h3>
                {awayRoster.length === 0 ? (
                  <div className="text-gray-400 text-center py-4">No players added</div>
                ) : (
                  <div className="space-y-2">
                    {awayRoster.map((r) => (
                      <div
                        key={r.id}
                        className="flex justify-between items-center bg-gray-700 rounded p-3"
                      >
                        <div>
                          <div className="font-semibold">{r.player.name}</div>
                          <div className="text-sm text-gray-400">
                            #{r.jersey_number_override || r.player.jersey_number}
                          </div>
                        </div>
                        <button
                          onClick={() => handleRemovePlayerFromRoster(r.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Add Player Modal */}
        {showAddPlayerModal && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                setShowAddPlayerModal(false);
              }
            }}
          >
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-md">
              <h2 className="text-2xl font-bold mb-4">Add Player to Roster</h2>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Team Side</label>
                <div className="flex gap-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      value="home"
                      checked={selectedTeamSide === 'home'}
                      onChange={(e) => setSelectedTeamSide(e.target.value as 'home' | 'away')}
                      className="mr-2"
                    />
                    {game.home_team}
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      value="away"
                      checked={selectedTeamSide === 'away'}
                      onChange={(e) => setSelectedTeamSide(e.target.value as 'home' | 'away')}
                      className="mr-2"
                    />
                    {game.away_team}
                  </label>
                </div>
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium mb-2">Select Player</label>
                {availablePlayers.length === 0 ? (
                  <div className="text-gray-400 text-center py-4">
                    All players have been added to the roster
                  </div>
                ) : (
                  <div className="max-h-64 overflow-y-auto space-y-2">
                    {availablePlayers.map((player) => (
                      <button
                        key={player.id}
                        onClick={() => handleAddPlayerToRoster(player.id, selectedTeamSide)}
                        className="w-full text-left bg-gray-700 hover:bg-gray-600 rounded p-3 transition-colors"
                      >
                        <div className="font-semibold">{player.name}</div>
                        <div className="text-sm text-gray-400">
                          #{player.jersey_number} • {player.team}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex justify-end">
                <button
                  onClick={() => setShowAddPlayerModal(false)}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
