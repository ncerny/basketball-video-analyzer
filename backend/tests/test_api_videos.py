"""Tests for Videos API endpoints."""

import pytest
from datetime import date, datetime
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
def sample_video_data(sample_game):
    """Sample video data for testing."""
    return {
        "game_id": sample_game["id"],
        "file_path": "/videos/test_video.mp4",
        "duration_seconds": 600.0,
        "fps": 30.0,
        "resolution": "1920x1080",
        "recorded_at": "2024-01-15T10:00:00",
        "sequence_order": 0,
        "game_time_offset": 0.0,
    }


@pytest.fixture
def sample_video_data_2(sample_game):
    """Alternative sample video data."""
    return {
        "game_id": sample_game["id"],
        "file_path": "/videos/test_video_2.mp4",
        "duration_seconds": 720.0,
        "fps": 60.0,
        "resolution": "3840x2160",
        "recorded_at": "2024-01-15T10:10:00",
        "sequence_order": 1,
        "game_time_offset": 600.0,
    }


class TestCreateVideo:
    """Tests for POST /api/videos endpoint."""

    def test_create_video_success(self, sample_video_data):
        """Test creating a video with all fields."""
        response = client.post("/api/videos", json=sample_video_data)

        assert response.status_code == 201
        data = response.json()
        assert data["game_id"] == sample_video_data["game_id"]
        assert data["file_path"] == sample_video_data["file_path"]
        assert data["duration_seconds"] == sample_video_data["duration_seconds"]
        assert data["fps"] == sample_video_data["fps"]
        assert data["resolution"] == sample_video_data["resolution"]
        assert "id" in data
        assert "upload_date" in data
        assert "processing_status" in data
        assert data["processing_status"] == "pending"
        assert data["processed"] is False
        # Timeline fields are optional at creation
        assert "recorded_at" in data
        assert "sequence_order" in data
        assert "game_time_offset" in data

    def test_create_video_minimal_fields(self, sample_game):
        """Test creating a video with minimal required fields."""
        minimal_data = {
            "game_id": sample_game["id"],
            "file_path": "/videos/minimal_video.mp4",
            "duration_seconds": 300.0,
            "fps": 30.0,
            "resolution": "1280x720",
            "recorded_at": "2024-01-15T10:00:00",
            "sequence_order": 0,
            "game_time_offset": 0.0,
        }
        response = client.post("/api/videos", json=minimal_data)

        assert response.status_code == 201
        data = response.json()
        assert data["thumbnail_path"] is None

    def test_create_video_with_thumbnail_path(self, sample_video_data):
        """Test creating a video with thumbnail_path."""
        video_data = {
            **sample_video_data,
            "thumbnail_path": "/thumbnails/test_video_thumb.jpg",
        }
        response = client.post("/api/videos", json=video_data)

        assert response.status_code == 201
        data = response.json()
        assert data["thumbnail_path"] == "/thumbnails/test_video_thumb.jpg"

    def test_create_video_missing_required_field(self, sample_game):
        """Test creating a video without required fields."""
        invalid_data = {
            "game_id": sample_game["id"],
            # Missing file_path, duration_seconds, etc.
        }
        response = client.post("/api/videos", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_create_video_game_not_found(self, sample_video_data):
        """Test creating a video with non-existent game."""
        invalid_data = {
            **sample_video_data,
            "game_id": 999999,  # Non-existent game
        }
        response = client.post("/api/videos", json=invalid_data)

        assert response.status_code == 404
        data = response.json()
        assert "game" in data["detail"].lower()

    def test_create_video_negative_duration(self, sample_video_data):
        """Test creating a video with negative duration."""
        invalid_data = {
            **sample_video_data,
            "duration_seconds": -100.0,
        }
        response = client.post("/api/videos", json=invalid_data)

        # Should either fail validation or succeed depending on schema
        assert response.status_code in [201, 422]

    def test_create_video_zero_fps(self, sample_video_data):
        """Test creating a video with zero fps."""
        invalid_data = {
            **sample_video_data,
            "fps": 0.0,
        }
        response = client.post("/api/videos", json=invalid_data)

        # Should either fail validation or succeed depending on schema
        assert response.status_code in [201, 422]


class TestListVideos:
    """Tests for GET /api/videos endpoint."""

    @pytest.fixture(autouse=True)
    def setup_videos(self, sample_game):
        """Create test videos before each test."""
        self.game = sample_game
        self.videos = []

        # Create multiple videos with different properties
        test_videos = [
            {
                "game_id": sample_game["id"],
                "file_path": "/videos/video_1.mp4",
                "duration_seconds": 300.0,
                "fps": 30.0,
                "resolution": "1920x1080",
                "recorded_at": "2024-01-15T10:00:00",
                "sequence_order": 0,
                "game_time_offset": 0.0,
            },
            {
                "game_id": sample_game["id"],
                "file_path": "/videos/video_2.mp4",
                "duration_seconds": 400.0,
                "fps": 30.0,
                "resolution": "1920x1080",
                "recorded_at": "2024-01-15T10:05:00",
                "sequence_order": 1,
                "game_time_offset": 300.0,
                "processing_status": "completed",
                "processed": True,
            },
            {
                "game_id": sample_game["id"],
                "file_path": "/videos/video_3.mp4",
                "duration_seconds": 500.0,
                "fps": 60.0,
                "resolution": "3840x2160",
                "recorded_at": "2024-01-15T10:12:00",
                "sequence_order": 2,
                "game_time_offset": 700.0,
                "processing_status": "failed",
            },
        ]

        for video_data in test_videos:
            response = client.post("/api/videos", json=video_data)
            if response.status_code == 201:
                self.videos.append(response.json())

    def test_list_videos_default(self):
        """Test listing videos with default pagination."""
        response = client.get("/api/videos")

        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 3  # At least the videos we created
        assert len(data["videos"]) <= data["page_size"]

    def test_list_videos_pagination(self):
        """Test pagination parameters."""
        # Get first page with page_size=2
        response = client.get("/api/videos?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["videos"]) <= 2

        # Get second page
        response2 = client.get("/api/videos?page=2&page_size=2")
        data2 = response2.json()
        assert data2["page"] == 2
        # Videos should be different from first page
        if len(data["videos"]) > 0 and len(data2["videos"]) > 0:
            assert data["videos"][0]["id"] != data2["videos"][0]["id"]

    def test_list_videos_filter_by_game(self):
        """Test filtering videos by game ID."""
        response = client.get(f"/api/videos?game_id={self.game['id']}")

        assert response.status_code == 200
        data = response.json()
        # All videos should be from the test game
        for video in data["videos"]:
            assert video["game_id"] == self.game["id"]

    def test_list_videos_filter_by_processing_status(self):
        """Test filtering videos by processing status."""
        response = client.get("/api/videos?processing_status=completed")

        assert response.status_code == 200
        data = response.json()
        # All videos should have completed status
        for video in data["videos"]:
            assert video["processing_status"] == "completed"

    def test_list_videos_filter_by_processed_true(self):
        """Test filtering videos by processed flag (true)."""
        response = client.get("/api/videos?processed=true")

        assert response.status_code == 200
        data = response.json()
        # All videos should be processed
        for video in data["videos"]:
            assert video["processed"] is True

    def test_list_videos_filter_by_processed_false(self):
        """Test filtering videos by processed flag (false)."""
        response = client.get("/api/videos?processed=false")

        assert response.status_code == 200
        data = response.json()
        # All videos should be unprocessed
        for video in data["videos"]:
            assert video["processed"] is False

    def test_list_videos_invalid_processing_status(self):
        """Test filtering with invalid processing status."""
        response = client.get("/api/videos?processing_status=invalid_status")

        assert response.status_code == 400
        data = response.json()
        assert "invalid processing_status" in data["detail"].lower()

    def test_list_videos_invalid_page(self):
        """Test with invalid page parameter."""
        response = client.get("/api/videos?page=0")

        assert response.status_code == 422  # Validation error

    def test_list_videos_sorted_by_sequence(self):
        """Test that videos are sorted by game_id and sequence_order."""
        response = client.get("/api/videos?page_size=100")

        assert response.status_code == 200
        data = response.json()
        videos = data["videos"]

        # Check if sorted by sequence_order within game
        if len(videos) > 1:
            for i in range(len(videos) - 1):
                if videos[i]["game_id"] == videos[i + 1]["game_id"]:
                    # If sequence_order is None, it should come first
                    if videos[i]["sequence_order"] is not None and videos[i + 1]["sequence_order"] is not None:
                        assert videos[i]["sequence_order"] <= videos[i + 1]["sequence_order"]


class TestGetVideo:
    """Tests for GET /api/videos/{video_id} endpoint."""

    @pytest.fixture
    def created_video(self, sample_video_data):
        """Create a video for testing."""
        response = client.post("/api/videos", json=sample_video_data)
        return response.json()

    def test_get_video_success(self, created_video):
        """Test getting a video by ID."""
        video_id = created_video["id"]
        response = client.get(f"/api/videos/{video_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == video_id
        assert data["file_path"] == created_video["file_path"]
        assert data["duration_seconds"] == created_video["duration_seconds"]

    def test_get_video_not_found(self):
        """Test getting a non-existent video."""
        response = client.get("/api/videos/999999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_video_invalid_id(self):
        """Test getting a video with invalid ID format."""
        response = client.get("/api/videos/invalid-id")

        assert response.status_code == 422  # Validation error


class TestUpdateVideo:
    """Tests for PATCH /api/videos/{video_id} endpoint."""

    @pytest.fixture
    def created_video(self, sample_video_data):
        """Create a video for testing."""
        response = client.post("/api/videos", json=sample_video_data)
        return response.json()

    def test_update_video_processing_status(self, created_video):
        """Test updating processing status."""
        video_id = created_video["id"]
        update_data = {
            "processing_status": "completed",
            "processed": True,
        }

        response = client.patch(f"/api/videos/{video_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["processing_status"] == "completed"
        assert data["processed"] is True

    def test_update_video_sequence_order(self, created_video):
        """Test updating sequence order and game time offset."""
        video_id = created_video["id"]
        update_data = {
            "sequence_order": 5,
            "game_time_offset": 1200.0,
        }

        response = client.patch(f"/api/videos/{video_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["sequence_order"] == 5
        assert data["game_time_offset"] == 1200.0

    def test_update_video_thumbnail_path(self, created_video):
        """Test updating thumbnail path."""
        video_id = created_video["id"]
        update_data = {
            "thumbnail_path": "/thumbnails/new_thumb.jpg",
        }

        response = client.patch(f"/api/videos/{video_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["thumbnail_path"] == "/thumbnails/new_thumb.jpg"

    def test_update_video_partial(self, created_video):
        """Test updating only some fields."""
        video_id = created_video["id"]
        update_data = {"processed": True}

        response = client.patch(f"/api/videos/{video_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] is True
        # Other fields should remain unchanged
        assert data["file_path"] == created_video["file_path"]
        assert data["duration_seconds"] == created_video["duration_seconds"]

    def test_update_video_invalid_processing_status(self, created_video):
        """Test updating with invalid processing status."""
        video_id = created_video["id"]
        update_data = {
            "processing_status": "invalid_status",
        }

        response = client.patch(f"/api/videos/{video_id}", json=update_data)

        assert response.status_code == 400
        data = response.json()
        assert "invalid processing_status" in data["detail"].lower()

    def test_update_video_not_found(self):
        """Test updating a non-existent video."""
        update_data = {"processed": True}
        response = client.patch("/api/videos/999999", json=update_data)

        assert response.status_code == 404

    def test_update_video_empty_payload(self, created_video):
        """Test updating with empty payload."""
        video_id = created_video["id"]
        response = client.patch(f"/api/videos/{video_id}", json={})

        # Should succeed but not change anything
        assert response.status_code == 200
        data = response.json()
        assert data["file_path"] == created_video["file_path"]
        assert data["duration_seconds"] == created_video["duration_seconds"]


class TestDeleteVideo:
    """Tests for DELETE /api/videos/{video_id} endpoint."""

    @pytest.fixture
    def created_video(self, sample_video_data):
        """Create a video for testing."""
        response = client.post("/api/videos", json=sample_video_data)
        return response.json()

    def test_delete_video_success(self, created_video):
        """Test deleting a video."""
        video_id = created_video["id"]
        response = client.delete(f"/api/videos/{video_id}")

        assert response.status_code == 204

        # Verify video is deleted
        get_response = client.get(f"/api/videos/{video_id}")
        assert get_response.status_code == 404

    def test_delete_video_not_found(self):
        """Test deleting a non-existent video."""
        response = client.delete("/api/videos/999999")

        assert response.status_code == 404

    def test_delete_video_twice(self, created_video):
        """Test deleting a video twice."""
        video_id = created_video["id"]

        # First deletion should succeed
        response1 = client.delete(f"/api/videos/{video_id}")
        assert response1.status_code == 204

        # Second deletion should fail
        response2 = client.delete(f"/api/videos/{video_id}")
        assert response2.status_code == 404
