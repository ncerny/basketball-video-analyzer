"""Tests for Game Rosters API endpoints."""

import pytest
from datetime import date
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_game():
    """Create a sample game for testing."""
    game_data = {
        "name": "Test Game",
        "date": str(date.today()),
        "home_team": "Warriors",
        "away_team": "Lakers",
    }
    response = client.post("/api/games", json=game_data)
    return response.json()


@pytest.fixture
def sample_player():
    """Create a sample player for testing."""
    player_data = {
        "name": "Stephen Curry",
        "jersey_number": 30,
        "team": "Warriors",
    }
    response = client.post("/api/players", json=player_data)
    return response.json()


@pytest.fixture
def sample_player_2():
    """Create another sample player for testing."""
    player_data = {
        "name": "LeBron James",
        "jersey_number": 23,
        "team": "Lakers",
    }
    response = client.post("/api/players", json=player_data)
    return response.json()


class TestCreateGameRoster:
    """Tests for POST /api/game-rosters endpoint."""

    def test_add_player_to_game_success(self, sample_game, sample_player):
        """Test adding a player to a game roster."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 201
        data = response.json()
        assert data["game_id"] == roster_data["game_id"]
        assert data["player_id"] == roster_data["player_id"]
        assert data["team_side"] == "home"
        assert data["jersey_number_override"] is None
        assert "id" in data

    def test_add_player_with_jersey_override(self, sample_game, sample_player):
        """Test adding a player with jersey number override."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
            "jersey_number_override": 7,
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 201
        data = response.json()
        assert data["jersey_number_override"] == 7

    def test_add_player_away_team(self, sample_game, sample_player):
        """Test adding a player to away team."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "away",
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 201
        data = response.json()
        assert data["team_side"] == "away"

    def test_add_player_duplicate(self, sample_game, sample_player):
        """Test adding the same player twice to the same game."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }

        # Add player first time
        response1 = client.post("/api/game-rosters", json=roster_data)
        assert response1.status_code == 201

        # Try to add same player again
        response2 = client.post("/api/game-rosters", json=roster_data)
        assert response2.status_code == 409  # Conflict
        data = response2.json()
        assert "already on the roster" in data["detail"].lower()

    def test_add_player_game_not_found(self, sample_player):
        """Test adding a player to non-existent game."""
        roster_data = {
            "game_id": 999999,  # Non-existent game
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 404
        data = response.json()
        assert "game" in data["detail"].lower()

    def test_add_player_player_not_found(self, sample_game):
        """Test adding non-existent player to a game."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": 999999,  # Non-existent player
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 404
        data = response.json()
        assert "player" in data["detail"].lower()

    def test_add_player_invalid_team_side(self, sample_game, sample_player):
        """Test adding a player with invalid team side."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "invalid_side",
        }
        response = client.post("/api/game-rosters", json=roster_data)

        assert response.status_code == 400
        data = response.json()
        assert "invalid team_side" in data["detail"].lower()

    def test_add_player_missing_required_field(self, sample_game, sample_player):
        """Test adding a player without required fields."""
        invalid_data = {
            "game_id": sample_game["id"],
            # Missing player_id and team_side
        }
        response = client.post("/api/game-rosters", json=invalid_data)

        assert response.status_code == 422  # Validation error


class TestListGameRosters:
    """Tests for GET /api/game-rosters endpoint."""

    @pytest.fixture(autouse=True)
    def setup_rosters(self, sample_game, sample_player, sample_player_2):
        """Create test roster entries before each test."""
        self.game = sample_game
        self.player1 = sample_player
        self.player2 = sample_player_2
        self.rosters = []

        # Add players to roster
        test_rosters = [
            {
                "game_id": sample_game["id"],
                "player_id": sample_player["id"],
                "team_side": "home",
                "jersey_number_override": 30,
            },
            {
                "game_id": sample_game["id"],
                "player_id": sample_player_2["id"],
                "team_side": "away",
                "jersey_number_override": 23,
            },
        ]

        for roster_data in test_rosters:
            response = client.post("/api/game-rosters", json=roster_data)
            if response.status_code == 201:
                self.rosters.append(response.json())

    def test_list_rosters_default(self):
        """Test listing roster entries with default pagination."""
        response = client.get("/api/game-rosters")

        assert response.status_code == 200
        data = response.json()
        assert "rosters" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 2  # At least the rosters we created
        assert len(data["rosters"]) <= data["page_size"]

    def test_list_rosters_filter_by_game(self):
        """Test filtering roster entries by game ID."""
        response = client.get(f"/api/game-rosters?game_id={self.game['id']}")

        assert response.status_code == 200
        data = response.json()
        # All rosters should be from the test game
        for roster in data["rosters"]:
            assert roster["game_id"] == self.game["id"]

    def test_list_rosters_filter_by_player(self):
        """Test filtering roster entries by player ID."""
        response = client.get(f"/api/game-rosters?player_id={self.player1['id']}")

        assert response.status_code == 200
        data = response.json()
        # All rosters should be for the specified player
        for roster in data["rosters"]:
            assert roster["player_id"] == self.player1["id"]

    def test_list_rosters_filter_by_team_side(self):
        """Test filtering roster entries by team side."""
        response = client.get("/api/game-rosters?team_side=home")

        assert response.status_code == 200
        data = response.json()
        # All rosters should be for home team
        for roster in data["rosters"]:
            assert roster["team_side"] == "home"

    def test_list_rosters_invalid_team_side(self):
        """Test filtering with invalid team side."""
        response = client.get("/api/game-rosters?team_side=invalid_side")

        assert response.status_code == 400
        data = response.json()
        assert "invalid team_side" in data["detail"].lower()

    def test_list_rosters_pagination(self):
        """Test pagination parameters."""
        # Get first page with page_size=1
        response = client.get("/api/game-rosters?page=1&page_size=1")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 1
        assert len(data["rosters"]) <= 1

    def test_list_rosters_invalid_page(self):
        """Test with invalid page parameter."""
        response = client.get("/api/game-rosters?page=0")

        assert response.status_code == 422  # Validation error


class TestGetGameRoster:
    """Tests for GET /api/game-rosters/{roster_id} endpoint."""

    @pytest.fixture
    def created_roster(self, sample_game, sample_player):
        """Create a roster entry for testing."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)
        return response.json()

    def test_get_roster_success(self, created_roster):
        """Test getting a roster entry by ID."""
        roster_id = created_roster["id"]
        response = client.get(f"/api/game-rosters/{roster_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == roster_id
        assert data["game_id"] == created_roster["game_id"]
        assert data["player_id"] == created_roster["player_id"]
        assert data["team_side"] == created_roster["team_side"]

    def test_get_roster_not_found(self):
        """Test getting a non-existent roster entry."""
        response = client.get("/api/game-rosters/999999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_roster_invalid_id(self):
        """Test getting a roster entry with invalid ID format."""
        response = client.get("/api/game-rosters/invalid-id")

        assert response.status_code == 422  # Validation error


class TestGetGameRosterWithDetails:
    """Tests for GET /api/game-rosters/with-details endpoint."""

    @pytest.fixture(autouse=True)
    def setup_rosters(self, sample_game, sample_player, sample_player_2):
        """Create test roster entries with player details."""
        self.game = sample_game

        # Add players to roster
        roster_data_1 = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        roster_data_2 = {
            "game_id": sample_game["id"],
            "player_id": sample_player_2["id"],
            "team_side": "away",
        }
        client.post("/api/game-rosters", json=roster_data_1)
        client.post("/api/game-rosters", json=roster_data_2)

    def test_get_roster_with_details_success(self):
        """Test getting roster with player details."""
        response = client.get(f"/api/game-rosters/with-details?game_id={self.game['id']}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2  # Two players on the roster

        # Check that player details are included
        for roster in data:
            assert "player_name" in roster
            assert "player_default_jersey" in roster
            assert "player_team" in roster
            assert "team_side" in roster

    def test_get_roster_with_details_game_not_found(self):
        """Test getting roster with details for non-existent game."""
        response = client.get("/api/game-rosters/with-details?game_id=999999")

        assert response.status_code == 404
        data = response.json()
        assert "game" in data["detail"].lower()

    def test_get_roster_with_details_missing_game_id(self):
        """Test getting roster without providing game_id."""
        response = client.get("/api/game-rosters/with-details")

        assert response.status_code == 422  # Validation error


class TestUpdateGameRoster:
    """Tests for PATCH /api/game-rosters/{roster_id} endpoint."""

    @pytest.fixture
    def created_roster(self, sample_game, sample_player):
        """Create a roster entry for testing."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)
        return response.json()

    def test_update_roster_team_side(self, created_roster):
        """Test updating team side."""
        roster_id = created_roster["id"]
        update_data = {
            "team_side": "away",
        }

        response = client.patch(f"/api/game-rosters/{roster_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["team_side"] == "away"

    def test_update_roster_jersey_override(self, created_roster):
        """Test updating jersey number override."""
        roster_id = created_roster["id"]
        update_data = {
            "jersey_number_override": 99,
        }

        response = client.patch(f"/api/game-rosters/{roster_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["jersey_number_override"] == 99

    def test_update_roster_all_fields(self, created_roster):
        """Test updating all fields."""
        roster_id = created_roster["id"]
        update_data = {
            "team_side": "away",
            "jersey_number_override": 77,
        }

        response = client.patch(f"/api/game-rosters/{roster_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["team_side"] == "away"
        assert data["jersey_number_override"] == 77

    def test_update_roster_invalid_team_side(self, created_roster):
        """Test updating with invalid team side."""
        roster_id = created_roster["id"]
        update_data = {
            "team_side": "invalid_side",
        }

        response = client.patch(f"/api/game-rosters/{roster_id}", json=update_data)

        assert response.status_code == 400
        data = response.json()
        assert "invalid team_side" in data["detail"].lower()

    def test_update_roster_not_found(self):
        """Test updating a non-existent roster entry."""
        update_data = {
            "team_side": "away",
        }
        response = client.patch("/api/game-rosters/999999", json=update_data)

        assert response.status_code == 404

    def test_update_roster_empty_payload(self, created_roster):
        """Test updating with empty payload."""
        roster_id = created_roster["id"]
        response = client.patch(f"/api/game-rosters/{roster_id}", json={})

        # Should succeed but not change anything
        assert response.status_code == 200
        data = response.json()
        assert data["team_side"] == created_roster["team_side"]


class TestDeleteGameRoster:
    """Tests for DELETE /api/game-rosters/{roster_id} endpoint."""

    @pytest.fixture
    def created_roster(self, sample_game, sample_player):
        """Create a roster entry for testing."""
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        response = client.post("/api/game-rosters", json=roster_data)
        return response.json()

    def test_delete_roster_success(self, created_roster):
        """Test deleting a roster entry."""
        roster_id = created_roster["id"]
        response = client.delete(f"/api/game-rosters/{roster_id}")

        assert response.status_code == 204

        # Verify roster entry is deleted
        get_response = client.get(f"/api/game-rosters/{roster_id}")
        assert get_response.status_code == 404

    def test_delete_roster_not_found(self):
        """Test deleting a non-existent roster entry."""
        response = client.delete("/api/game-rosters/999999")

        assert response.status_code == 404

    def test_delete_roster_twice(self, created_roster):
        """Test deleting a roster entry twice."""
        roster_id = created_roster["id"]

        # First deletion should succeed
        response1 = client.delete(f"/api/game-rosters/{roster_id}")
        assert response1.status_code == 204

        # Second deletion should fail
        response2 = client.delete(f"/api/game-rosters/{roster_id}")
        assert response2.status_code == 404

    def test_delete_roster_player_can_be_re_added(self, sample_game, sample_player):
        """Test that a player can be re-added after removal."""
        # Add player to roster
        roster_data = {
            "game_id": sample_game["id"],
            "player_id": sample_player["id"],
            "team_side": "home",
        }
        create_response = client.post("/api/game-rosters", json=roster_data)
        roster_id = create_response.json()["id"]

        # Delete roster entry
        delete_response = client.delete(f"/api/game-rosters/{roster_id}")
        assert delete_response.status_code == 204

        # Re-add same player (should succeed)
        re_add_response = client.post("/api/game-rosters", json=roster_data)
        assert re_add_response.status_code == 201
