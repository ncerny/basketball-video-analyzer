"""Tests for Annotations API endpoints."""

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
def sample_video(sample_game):
    """Create a sample video for testing."""
    video_data = {
        "game_id": sample_game["id"],
        "filename": "test_video.mp4",
        "file_path": "/videos/test_video.mp4",
        "duration_seconds": 600.0,
        "fps": 30.0,
        "resolution": "1920x1080",
        "file_size_bytes": 104857600,
        "recorded_at": "2024-01-15T10:00:00",
        "sequence_order": 0,
        "game_time_offset": 0.0,
    }
    response = client.post("/api/videos", json=video_data)
    return response.json()


@pytest.fixture
def sample_annotation_data(sample_game):
    """Sample annotation data for testing."""
    return {
        "game_id": sample_game["id"],
        "game_timestamp_start": 10.0,
        "game_timestamp_end": 20.0,
        "annotation_type": "play",
        "confidence_score": 0.95,
        "verified": False,
        "created_by": "ai",
    }


class TestCreateAnnotation:
    """Tests for POST /api/annotations endpoint."""

    def test_create_annotation_success(self, sample_annotation_data):
        """Test creating an annotation with all fields."""
        response = client.post("/api/annotations", json=sample_annotation_data)

        assert response.status_code == 201
        data = response.json()
        assert data["game_id"] == sample_annotation_data["game_id"]
        assert data["game_timestamp_start"] == sample_annotation_data["game_timestamp_start"]
        assert data["game_timestamp_end"] == sample_annotation_data["game_timestamp_end"]
        assert data["annotation_type"] == sample_annotation_data["annotation_type"]
        assert data["confidence_score"] == sample_annotation_data["confidence_score"]
        assert data["verified"] == sample_annotation_data["verified"]
        assert data["created_by"] == sample_annotation_data["created_by"]
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "video_links" in data

    def test_create_annotation_with_videos(self, sample_annotation_data, sample_video):
        """Test creating an annotation with video links."""
        annotation_data = {
            **sample_annotation_data,
            "video_ids": [sample_video["id"]],
        }
        response = client.post("/api/annotations", json=annotation_data)

        assert response.status_code == 201
        data = response.json()
        assert len(data["video_links"]) == 1
        assert data["video_links"][0]["video_id"] == sample_video["id"]

    def test_create_annotation_event_type(self, sample_annotation_data):
        """Test creating an annotation with event type."""
        annotation_data = {
            **sample_annotation_data,
            "annotation_type": "event",
        }
        response = client.post("/api/annotations", json=annotation_data)

        assert response.status_code == 201
        data = response.json()
        assert data["annotation_type"] == "event"

    def test_create_annotation_note_type(self, sample_annotation_data):
        """Test creating an annotation with note type."""
        annotation_data = {
            **sample_annotation_data,
            "annotation_type": "note",
        }
        response = client.post("/api/annotations", json=annotation_data)

        assert response.status_code == 201
        data = response.json()
        assert data["annotation_type"] == "note"

    def test_create_annotation_user_created(self, sample_annotation_data):
        """Test creating a user-created annotation."""
        annotation_data = {
            **sample_annotation_data,
            "created_by": "user",
            "verified": True,
        }
        response = client.post("/api/annotations", json=annotation_data)

        assert response.status_code == 201
        data = response.json()
        assert data["created_by"] == "user"
        assert data["verified"] is True

    def test_create_annotation_missing_required_field(self, sample_game):
        """Test creating an annotation without required fields."""
        invalid_data = {
            "game_id": sample_game["id"],
            "game_timestamp_start": 10.0,
            # Missing game_timestamp_end, annotation_type, created_by
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_create_annotation_invalid_type(self, sample_annotation_data):
        """Test creating an annotation with invalid type."""
        invalid_data = {
            **sample_annotation_data,
            "annotation_type": "invalid_type",
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 400
        data = response.json()
        assert "invalid annotation_type" in data["detail"].lower()

    def test_create_annotation_invalid_created_by(self, sample_annotation_data):
        """Test creating an annotation with invalid created_by."""
        invalid_data = {
            **sample_annotation_data,
            "created_by": "invalid_creator",
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 400
        data = response.json()
        assert "invalid created_by" in data["detail"].lower()

    def test_create_annotation_invalid_timestamps(self, sample_annotation_data):
        """Test creating an annotation with end <= start."""
        invalid_data = {
            **sample_annotation_data,
            "game_timestamp_start": 20.0,
            "game_timestamp_end": 10.0,  # End before start
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 400
        data = response.json()
        assert "timestamp" in data["detail"].lower()

    def test_create_annotation_equal_timestamps(self, sample_annotation_data):
        """Test creating an annotation with equal timestamps."""
        invalid_data = {
            **sample_annotation_data,
            "game_timestamp_start": 10.0,
            "game_timestamp_end": 10.0,  # Equal timestamps
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 400

    def test_create_annotation_game_not_found(self, sample_annotation_data):
        """Test creating an annotation with non-existent game."""
        invalid_data = {
            **sample_annotation_data,
            "game_id": 999999,  # Non-existent game
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 404
        data = response.json()
        assert "game" in data["detail"].lower()

    def test_create_annotation_video_not_found(self, sample_annotation_data):
        """Test creating an annotation with non-existent video."""
        invalid_data = {
            **sample_annotation_data,
            "video_ids": [999999],  # Non-existent video
        }
        response = client.post("/api/annotations", json=invalid_data)

        assert response.status_code == 404
        data = response.json()
        assert "video" in data["detail"].lower()


class TestListAnnotations:
    """Tests for GET /api/annotations endpoint."""

    @pytest.fixture(autouse=True)
    def setup_annotations(self, sample_game):
        """Create test annotations before each test."""
        self.game = sample_game
        self.annotations = []

        # Create multiple annotations with different properties
        test_annotations = [
            {
                "game_id": sample_game["id"],
                "game_timestamp_start": 10.0,
                "game_timestamp_end": 20.0,
                "annotation_type": "play",
                "confidence_score": 0.95,
                "verified": True,
                "created_by": "user",
            },
            {
                "game_id": sample_game["id"],
                "game_timestamp_start": 30.0,
                "game_timestamp_end": 40.0,
                "annotation_type": "event",
                "confidence_score": 0.85,
                "verified": False,
                "created_by": "ai",
            },
            {
                "game_id": sample_game["id"],
                "game_timestamp_start": 50.0,
                "game_timestamp_end": 60.0,
                "annotation_type": "note",
                "confidence_score": 1.0,
                "verified": True,
                "created_by": "user",
            },
        ]

        for annotation_data in test_annotations:
            response = client.post("/api/annotations", json=annotation_data)
            if response.status_code == 201:
                self.annotations.append(response.json())

    def test_list_annotations_default(self):
        """Test listing annotations with default pagination."""
        response = client.get("/api/annotations")

        assert response.status_code == 200
        data = response.json()
        assert "annotations" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 3  # At least the annotations we created
        assert len(data["annotations"]) <= data["page_size"]

    def test_list_annotations_pagination(self):
        """Test pagination parameters."""
        # Get first page with page_size=2
        response = client.get("/api/annotations?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["annotations"]) <= 2

        # Get second page
        response2 = client.get("/api/annotations?page=2&page_size=2")
        data2 = response2.json()
        assert data2["page"] == 2
        # Annotations should be different from first page
        if len(data["annotations"]) > 0 and len(data2["annotations"]) > 0:
            assert data["annotations"][0]["id"] != data2["annotations"][0]["id"]

    def test_list_annotations_filter_by_game(self):
        """Test filtering annotations by game ID."""
        response = client.get(f"/api/annotations?game_id={self.game['id']}")

        assert response.status_code == 200
        data = response.json()
        # All annotations should be from the test game
        for annotation in data["annotations"]:
            assert annotation["game_id"] == self.game["id"]

    def test_list_annotations_filter_by_type(self):
        """Test filtering annotations by type."""
        response = client.get("/api/annotations?annotation_type=play")

        assert response.status_code == 200
        data = response.json()
        # All annotations should be of type 'play'
        for annotation in data["annotations"]:
            assert annotation["annotation_type"] == "play"

    def test_list_annotations_filter_by_created_by(self):
        """Test filtering annotations by creator."""
        response = client.get("/api/annotations?created_by=ai")

        assert response.status_code == 200
        data = response.json()
        # All annotations should be AI-created
        for annotation in data["annotations"]:
            assert annotation["created_by"] == "ai"

    def test_list_annotations_filter_by_verified(self):
        """Test filtering annotations by verified status."""
        response = client.get("/api/annotations?verified=true")

        assert response.status_code == 200
        data = response.json()
        # All annotations should be verified
        for annotation in data["annotations"]:
            assert annotation["verified"] is True

    def test_list_annotations_filter_by_unverified(self):
        """Test filtering annotations by unverified status."""
        response = client.get("/api/annotations?verified=false")

        assert response.status_code == 200
        data = response.json()
        # All annotations should be unverified
        for annotation in data["annotations"]:
            assert annotation["verified"] is False

    def test_list_annotations_invalid_type_filter(self):
        """Test filtering with invalid annotation_type."""
        response = client.get("/api/annotations?annotation_type=invalid_type")

        assert response.status_code == 400
        data = response.json()
        assert "invalid annotation_type" in data["detail"].lower()

    def test_list_annotations_invalid_created_by_filter(self):
        """Test filtering with invalid created_by."""
        response = client.get("/api/annotations?created_by=invalid_creator")

        assert response.status_code == 400
        data = response.json()
        assert "invalid created_by" in data["detail"].lower()

    def test_list_annotations_invalid_page(self):
        """Test with invalid page parameter."""
        response = client.get("/api/annotations?page=0")

        assert response.status_code == 422  # Validation error


class TestGetAnnotation:
    """Tests for GET /api/annotations/{annotation_id} endpoint."""

    @pytest.fixture
    def created_annotation(self, sample_annotation_data):
        """Create an annotation for testing."""
        response = client.post("/api/annotations", json=sample_annotation_data)
        return response.json()

    def test_get_annotation_success(self, created_annotation):
        """Test getting an annotation by ID."""
        annotation_id = created_annotation["id"]
        response = client.get(f"/api/annotations/{annotation_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == annotation_id
        assert data["game_id"] == created_annotation["game_id"]
        assert data["annotation_type"] == created_annotation["annotation_type"]

    def test_get_annotation_with_video_links(self, sample_annotation_data, sample_video):
        """Test getting an annotation with video links."""
        annotation_data = {
            **sample_annotation_data,
            "video_ids": [sample_video["id"]],
        }
        create_response = client.post("/api/annotations", json=annotation_data)
        created = create_response.json()

        response = client.get(f"/api/annotations/{created['id']}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["video_links"]) == 1
        assert data["video_links"][0]["video_id"] == sample_video["id"]

    def test_get_annotation_not_found(self):
        """Test getting a non-existent annotation."""
        response = client.get("/api/annotations/999999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_annotation_invalid_id(self):
        """Test getting an annotation with invalid ID format."""
        response = client.get("/api/annotations/invalid-id")

        assert response.status_code == 422  # Validation error


class TestUpdateAnnotation:
    """Tests for PATCH /api/annotations/{annotation_id} endpoint."""

    @pytest.fixture
    def created_annotation(self, sample_annotation_data):
        """Create an annotation for testing."""
        response = client.post("/api/annotations", json=sample_annotation_data)
        return response.json()

    def test_update_annotation_all_fields(self, created_annotation):
        """Test updating all fields of an annotation."""
        annotation_id = created_annotation["id"]
        update_data = {
            "game_timestamp_start": 15.0,
            "game_timestamp_end": 25.0,
            "annotation_type": "event",
            "confidence_score": 0.99,
            "verified": True,
            "created_by": "user",
        }

        response = client.patch(f"/api/annotations/{annotation_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["game_timestamp_start"] == update_data["game_timestamp_start"]
        assert data["game_timestamp_end"] == update_data["game_timestamp_end"]
        assert data["annotation_type"] == update_data["annotation_type"]
        assert data["confidence_score"] == update_data["confidence_score"]
        assert data["verified"] == update_data["verified"]
        assert data["created_by"] == update_data["created_by"]

    def test_update_annotation_partial(self, created_annotation):
        """Test updating only some fields."""
        annotation_id = created_annotation["id"]
        update_data = {"verified": True}

        response = client.patch(f"/api/annotations/{annotation_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["verified"] is True
        # Other fields should remain unchanged
        assert data["annotation_type"] == created_annotation["annotation_type"]
        assert data["confidence_score"] == created_annotation["confidence_score"]

    def test_update_annotation_with_video_links(self, created_annotation, sample_video):
        """Test updating annotation with video links."""
        annotation_id = created_annotation["id"]
        update_data = {
            "video_ids": [sample_video["id"]],
        }

        response = client.patch(f"/api/annotations/{annotation_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["video_links"]) == 1
        assert data["video_links"][0]["video_id"] == sample_video["id"]

    def test_update_annotation_invalid_timestamps(self, created_annotation):
        """Test updating with invalid timestamps."""
        annotation_id = created_annotation["id"]
        update_data = {
            "game_timestamp_start": 30.0,
            "game_timestamp_end": 20.0,  # End before start
        }

        response = client.patch(f"/api/annotations/{annotation_id}", json=update_data)

        assert response.status_code == 400
        data = response.json()
        assert "timestamp" in data["detail"].lower()

    def test_update_annotation_not_found(self):
        """Test updating a non-existent annotation."""
        update_data = {"verified": True}
        response = client.patch("/api/annotations/999999", json=update_data)

        assert response.status_code == 404

    def test_update_annotation_empty_payload(self, created_annotation):
        """Test updating with empty payload."""
        annotation_id = created_annotation["id"]
        response = client.patch(f"/api/annotations/{annotation_id}", json={})

        # Should succeed but not change anything
        assert response.status_code == 200
        data = response.json()
        assert data["annotation_type"] == created_annotation["annotation_type"]


class TestDeleteAnnotation:
    """Tests for DELETE /api/annotations/{annotation_id} endpoint."""

    @pytest.fixture
    def created_annotation(self, sample_annotation_data):
        """Create an annotation for testing."""
        response = client.post("/api/annotations", json=sample_annotation_data)
        return response.json()

    def test_delete_annotation_success(self, created_annotation):
        """Test deleting an annotation."""
        annotation_id = created_annotation["id"]
        response = client.delete(f"/api/annotations/{annotation_id}")

        assert response.status_code == 204

        # Verify annotation is deleted
        get_response = client.get(f"/api/annotations/{annotation_id}")
        assert get_response.status_code == 404

    def test_delete_annotation_with_video_links(self, sample_annotation_data, sample_video):
        """Test deleting an annotation with video links."""
        annotation_data = {
            **sample_annotation_data,
            "video_ids": [sample_video["id"]],
        }
        create_response = client.post("/api/annotations", json=annotation_data)
        created = create_response.json()

        response = client.delete(f"/api/annotations/{created['id']}")

        assert response.status_code == 204

    def test_delete_annotation_not_found(self):
        """Test deleting a non-existent annotation."""
        response = client.delete("/api/annotations/999999")

        assert response.status_code == 404

    def test_delete_annotation_twice(self, created_annotation):
        """Test deleting an annotation twice."""
        annotation_id = created_annotation["id"]

        # First deletion should succeed
        response1 = client.delete(f"/api/annotations/{annotation_id}")
        assert response1.status_code == 204

        # Second deletion should fail
        response2 = client.delete(f"/api/annotations/{annotation_id}")
        assert response2.status_code == 404
