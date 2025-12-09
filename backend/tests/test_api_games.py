"""Tests for Games API endpoints."""

import pytest
from datetime import date, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_game_data():
    """Sample game data for testing."""
    return {
        "name": "Warriors vs Lakers",
        "date": str(date.today()),
        "home_team": "Warriors",
        "away_team": "Lakers",
        "location": "Chase Center",
    }


@pytest.fixture
def sample_game_data_minimal():
    """Minimal game data for testing."""
    return {
        "name": "Test Game",
        "date": str(date.today()),
        "home_team": "Team A",
        "away_team": "Team B",
    }


class TestCreateGame:
    """Tests for POST /api/games endpoint."""

    def test_create_game_success(self, sample_game_data):
        """Test creating a game with all fields."""
        response = client.post("/api/games", json=sample_game_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_game_data["name"]
        assert data["date"] == sample_game_data["date"]
        assert data["home_team"] == sample_game_data["home_team"]
        assert data["away_team"] == sample_game_data["away_team"]
        assert data["location"] == sample_game_data["location"]
        assert "id" in data
        assert "created_at" in data

    def test_create_game_minimal(self, sample_game_data_minimal):
        """Test creating a game with minimal required fields."""
        response = client.post("/api/games", json=sample_game_data_minimal)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_game_data_minimal["name"]
        assert data["location"] is None

    def test_create_game_missing_required_field(self):
        """Test creating a game without required fields."""
        invalid_data = {
            "name": "Test Game",
            "home_team": "Team A",
            # Missing date and away_team
        }
        response = client.post("/api/games", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_create_game_invalid_date_format(self):
        """Test creating a game with invalid date format."""
        invalid_data = {
            "name": "Test Game",
            "date": "not-a-date",
            "home_team": "Team A",
            "away_team": "Team B",
        }
        response = client.post("/api/games", json=invalid_data)

        assert response.status_code == 422  # Validation error


class TestListGames:
    """Tests for GET /api/games endpoint."""

    @pytest.fixture(autouse=True)
    def setup_games(self):
        """Create test games before each test."""
        # Create multiple games with different dates and teams
        self.games = []
        for i in range(5):
            game_data = {
                "name": f"Game {i+1}",
                "date": str(date.today() - timedelta(days=i)),
                "home_team": "Warriors" if i % 2 == 0 else "Lakers",
                "away_team": "Celtics" if i % 2 == 0 else "Heat",
                "location": f"Venue {i+1}",
            }
            response = client.post("/api/games", json=game_data)
            self.games.append(response.json())

    def test_list_games_default(self):
        """Test listing games with default pagination."""
        response = client.get("/api/games")

        assert response.status_code == 200
        data = response.json()
        assert "games" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 5  # At least the games we created
        assert len(data["games"]) <= data["page_size"]

    def test_list_games_pagination(self):
        """Test pagination parameters."""
        # Get first page with page_size=2
        response = client.get("/api/games?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["games"]) <= 2

        # Get second page
        response2 = client.get("/api/games?page=2&page_size=2")
        data2 = response2.json()
        assert data2["page"] == 2
        # Games should be different from first page
        if len(data["games"]) > 0 and len(data2["games"]) > 0:
            assert data["games"][0]["id"] != data2["games"][0]["id"]

    def test_list_games_filter_by_team(self):
        """Test filtering games by team."""
        response = client.get("/api/games?team=Warriors")

        assert response.status_code == 200
        data = response.json()
        # All games should have Warriors as home or away team
        for game in data["games"]:
            assert "Warriors" in game["home_team"] or "Warriors" in game["away_team"]

    def test_list_games_filter_by_date_range(self):
        """Test filtering games by date range."""
        start_date = str(date.today() - timedelta(days=2))
        end_date = str(date.today())

        response = client.get(f"/api/games?date_from={start_date}&date_to={end_date}")

        assert response.status_code == 200
        data = response.json()
        # All games should be within the date range
        for game in data["games"]:
            game_date = date.fromisoformat(game["date"])
            assert date.fromisoformat(start_date) <= game_date <= date.fromisoformat(end_date)

    def test_list_games_search_by_name(self):
        """Test searching games by name."""
        response = client.get("/api/games?search=Game 1")

        assert response.status_code == 200
        data = response.json()
        # All games should match the search term
        for game in data["games"]:
            assert "Game 1" in game["name"]

    def test_list_games_invalid_page(self):
        """Test with invalid page parameter."""
        response = client.get("/api/games?page=0")

        assert response.status_code == 422  # Validation error


class TestGetGame:
    """Tests for GET /api/games/{game_id} endpoint."""

    @pytest.fixture
    def created_game(self, sample_game_data):
        """Create a game for testing."""
        response = client.post("/api/games", json=sample_game_data)
        return response.json()

    def test_get_game_success(self, created_game):
        """Test getting a game by ID."""
        game_id = created_game["id"]
        response = client.get(f"/api/games/{game_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == game_id
        assert data["name"] == created_game["name"]

    def test_get_game_not_found(self):
        """Test getting a non-existent game."""
        response = client.get("/api/games/999999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_game_invalid_id(self):
        """Test getting a game with invalid ID format."""
        response = client.get("/api/games/invalid-id")

        assert response.status_code == 422  # Validation error


class TestUpdateGame:
    """Tests for PATCH /api/games/{game_id} endpoint."""

    @pytest.fixture
    def created_game(self, sample_game_data):
        """Create a game for testing."""
        response = client.post("/api/games", json=sample_game_data)
        return response.json()

    def test_update_game_all_fields(self, created_game):
        """Test updating all fields of a game."""
        game_id = created_game["id"]
        update_data = {
            "name": "Updated Game Name",
            "date": str(date.today() + timedelta(days=1)),
            "home_team": "Updated Home",
            "away_team": "Updated Away",
            "location": "Updated Location",
        }

        response = client.patch(f"/api/games/{game_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["date"] == update_data["date"]
        assert data["home_team"] == update_data["home_team"]
        assert data["away_team"] == update_data["away_team"]
        assert data["location"] == update_data["location"]

    def test_update_game_partial(self, created_game):
        """Test updating only some fields."""
        game_id = created_game["id"]
        update_data = {"name": "Partially Updated"}

        response = client.patch(f"/api/games/{game_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        # Other fields should remain unchanged
        assert data["home_team"] == created_game["home_team"]
        assert data["away_team"] == created_game["away_team"]

    def test_update_game_not_found(self):
        """Test updating a non-existent game."""
        update_data = {"name": "Updated Name"}
        response = client.patch("/api/games/999999", json=update_data)

        assert response.status_code == 404

    def test_update_game_empty_payload(self, created_game):
        """Test updating with empty payload."""
        game_id = created_game["id"]
        response = client.patch(f"/api/games/{game_id}", json={})

        # Should succeed but not change anything
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == created_game["name"]


class TestDeleteGame:
    """Tests for DELETE /api/games/{game_id} endpoint."""

    @pytest.fixture
    def created_game(self, sample_game_data):
        """Create a game for testing."""
        response = client.post("/api/games", json=sample_game_data)
        return response.json()

    def test_delete_game_success(self, created_game):
        """Test deleting a game."""
        game_id = created_game["id"]
        response = client.delete(f"/api/games/{game_id}")

        assert response.status_code == 204

        # Verify game is deleted
        get_response = client.get(f"/api/games/{game_id}")
        assert get_response.status_code == 404

    def test_delete_game_not_found(self):
        """Test deleting a non-existent game."""
        response = client.delete("/api/games/999999")

        assert response.status_code == 404

    def test_delete_game_twice(self, created_game):
        """Test deleting a game twice."""
        game_id = created_game["id"]

        # First deletion should succeed
        response1 = client.delete(f"/api/games/{game_id}")
        assert response1.status_code == 204

        # Second deletion should fail
        response2 = client.delete(f"/api/games/{game_id}")
        assert response2.status_code == 404
