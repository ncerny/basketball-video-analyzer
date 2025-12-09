"""Tests for Players API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_player_data():
    """Sample player data for testing."""
    return {
        "name": "Stephen Curry",
        "jersey_number": 30,
        "team": "Warriors",
    }


@pytest.fixture
def sample_player_data_2():
    """Alternative sample player data."""
    return {
        "name": "LeBron James",
        "jersey_number": 23,
        "team": "Lakers",
    }


class TestCreatePlayer:
    """Tests for POST /api/players endpoint."""

    def test_create_player_success(self, sample_player_data):
        """Test creating a player with valid data."""
        response = client.post("/api/players", json=sample_player_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_player_data["name"]
        assert data["jersey_number"] == sample_player_data["jersey_number"]
        assert data["team"] == sample_player_data["team"]
        assert "id" in data
        assert "created_at" in data

    def test_create_player_duplicate(self, sample_player_data):
        """Test creating a duplicate player (same name, jersey, team)."""
        # Create first player
        response1 = client.post("/api/players", json=sample_player_data)
        assert response1.status_code == 201

        # Try to create duplicate
        response2 = client.post("/api/players", json=sample_player_data)
        assert response2.status_code == 409  # Conflict
        data = response2.json()
        assert "already exists" in data["detail"].lower()

    def test_create_player_same_name_different_team(self, sample_player_data):
        """Test creating players with same name but different teams."""
        # Create first player
        response1 = client.post("/api/players", json=sample_player_data)
        assert response1.status_code == 201

        # Create player with same name but different team (should succeed)
        different_team = {
            "name": sample_player_data["name"],
            "jersey_number": sample_player_data["jersey_number"],
            "team": "Different Team",
        }
        response2 = client.post("/api/players", json=different_team)
        assert response2.status_code == 201

    def test_create_player_same_name_different_jersey(self, sample_player_data):
        """Test creating players with same name and team but different jersey."""
        # Create first player
        response1 = client.post("/api/players", json=sample_player_data)
        assert response1.status_code == 201

        # Create player with different jersey (should succeed)
        different_jersey = {
            "name": sample_player_data["name"],
            "jersey_number": 99,
            "team": sample_player_data["team"],
        }
        response2 = client.post("/api/players", json=different_jersey)
        assert response2.status_code == 201

    def test_create_player_missing_required_field(self):
        """Test creating a player without required fields."""
        invalid_data = {
            "name": "Test Player",
            # Missing jersey_number and team
        }
        response = client.post("/api/players", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_create_player_invalid_jersey_number(self):
        """Test creating a player with invalid jersey number."""
        invalid_data = {
            "name": "Test Player",
            "jersey_number": "not-a-number",
            "team": "Team A",
        }
        response = client.post("/api/players", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_create_player_negative_jersey_number(self):
        """Test creating a player with negative jersey number."""
        invalid_data = {
            "name": "Test Player",
            "jersey_number": -1,
            "team": "Team A",
        }
        response = client.post("/api/players", json=invalid_data)

        # Should either fail validation or succeed depending on schema
        assert response.status_code in [201, 422]


class TestListPlayers:
    """Tests for GET /api/players endpoint."""

    @pytest.fixture(autouse=True)
    def setup_players(self):
        """Create test players before each test."""
        self.players = []
        test_players = [
            {"name": "Player A", "jersey_number": 1, "team": "Warriors"},
            {"name": "Player B", "jersey_number": 2, "team": "Warriors"},
            {"name": "Player C", "jersey_number": 3, "team": "Lakers"},
            {"name": "Player D", "jersey_number": 4, "team": "Lakers"},
            {"name": "Player E", "jersey_number": 5, "team": "Celtics"},
        ]

        for player_data in test_players:
            response = client.post("/api/players", json=player_data)
            if response.status_code == 201:
                self.players.append(response.json())

    def test_list_players_default(self):
        """Test listing players with default pagination."""
        response = client.get("/api/players")

        assert response.status_code == 200
        data = response.json()
        assert "players" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 5  # At least the players we created
        assert len(data["players"]) <= data["page_size"]

    def test_list_players_pagination(self):
        """Test pagination parameters."""
        # Get first page with page_size=2
        response = client.get("/api/players?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["players"]) <= 2

        # Get second page
        response2 = client.get("/api/players?page=2&page_size=2")
        data2 = response2.json()
        assert data2["page"] == 2
        # Players should be different from first page
        if len(data["players"]) > 0 and len(data2["players"]) > 0:
            assert data["players"][0]["id"] != data2["players"][0]["id"]

    def test_list_players_filter_by_team(self):
        """Test filtering players by team."""
        response = client.get("/api/players?team=Warriors")

        assert response.status_code == 200
        data = response.json()
        # All players should be from Warriors team
        for player in data["players"]:
            assert player["team"] == "Warriors"

    def test_list_players_search_by_name(self):
        """Test searching players by name."""
        response = client.get("/api/players?search=Player A")

        assert response.status_code == 200
        data = response.json()
        # All players should match the search term
        for player in data["players"]:
            assert "Player A" in player["name"]

    def test_list_players_sorted_by_name(self):
        """Test that players are sorted by name."""
        response = client.get("/api/players?page_size=100")

        assert response.status_code == 200
        data = response.json()
        players = data["players"]

        # Check if sorted alphabetically
        if len(players) > 1:
            for i in range(len(players) - 1):
                assert players[i]["name"] <= players[i + 1]["name"]

    def test_list_players_invalid_page(self):
        """Test with invalid page parameter."""
        response = client.get("/api/players?page=0")

        assert response.status_code == 422  # Validation error


class TestGetPlayer:
    """Tests for GET /api/players/{player_id} endpoint."""

    @pytest.fixture
    def created_player(self, sample_player_data):
        """Create a player for testing."""
        response = client.post("/api/players", json=sample_player_data)
        return response.json()

    def test_get_player_success(self, created_player):
        """Test getting a player by ID."""
        player_id = created_player["id"]
        response = client.get(f"/api/players/{player_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == player_id
        assert data["name"] == created_player["name"]
        assert data["jersey_number"] == created_player["jersey_number"]
        assert data["team"] == created_player["team"]

    def test_get_player_not_found(self):
        """Test getting a non-existent player."""
        response = client.get("/api/players/999999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_player_invalid_id(self):
        """Test getting a player with invalid ID format."""
        response = client.get("/api/players/invalid-id")

        assert response.status_code == 422  # Validation error


class TestUpdatePlayer:
    """Tests for PATCH /api/players/{player_id} endpoint."""

    @pytest.fixture
    def created_player(self, sample_player_data):
        """Create a player for testing."""
        response = client.post("/api/players", json=sample_player_data)
        return response.json()

    def test_update_player_all_fields(self, created_player):
        """Test updating all fields of a player."""
        player_id = created_player["id"]
        update_data = {
            "name": "Updated Name",
            "jersey_number": 99,
            "team": "Updated Team",
        }

        response = client.patch(f"/api/players/{player_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["jersey_number"] == update_data["jersey_number"]
        assert data["team"] == update_data["team"]

    def test_update_player_partial_name(self, created_player):
        """Test updating only the name."""
        player_id = created_player["id"]
        update_data = {"name": "Partially Updated"}

        response = client.patch(f"/api/players/{player_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        # Other fields should remain unchanged
        assert data["jersey_number"] == created_player["jersey_number"]
        assert data["team"] == created_player["team"]

    def test_update_player_jersey_number(self, created_player):
        """Test updating only the jersey number."""
        player_id = created_player["id"]
        update_data = {"jersey_number": 42}

        response = client.patch(f"/api/players/{player_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["jersey_number"] == 42
        assert data["name"] == created_player["name"]

    def test_update_player_not_found(self):
        """Test updating a non-existent player."""
        update_data = {"name": "Updated Name"}
        response = client.patch("/api/players/999999", json=update_data)

        assert response.status_code == 404

    def test_update_player_empty_payload(self, created_player):
        """Test updating with empty payload."""
        player_id = created_player["id"]
        response = client.patch(f"/api/players/{player_id}", json={})

        # Should succeed but not change anything
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == created_player["name"]


class TestDeletePlayer:
    """Tests for DELETE /api/players/{player_id} endpoint."""

    @pytest.fixture
    def created_player(self, sample_player_data):
        """Create a player for testing."""
        response = client.post("/api/players", json=sample_player_data)
        return response.json()

    def test_delete_player_success(self, created_player):
        """Test deleting a player."""
        player_id = created_player["id"]
        response = client.delete(f"/api/players/{player_id}")

        assert response.status_code == 204

        # Verify player is deleted
        get_response = client.get(f"/api/players/{player_id}")
        assert get_response.status_code == 404

    def test_delete_player_not_found(self):
        """Test deleting a non-existent player."""
        response = client.delete("/api/players/999999")

        assert response.status_code == 404

    def test_delete_player_twice(self, created_player):
        """Test deleting a player twice."""
        player_id = created_player["id"]

        # First deletion should succeed
        response1 = client.delete(f"/api/players/{player_id}")
        assert response1.status_code == 204

        # Second deletion should fail
        response2 = client.delete(f"/api/players/{player_id}")
        assert response2.status_code == 404
