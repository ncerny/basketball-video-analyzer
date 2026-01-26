"""Tests for detection API endpoints."""

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
        "file_path": "/videos/test_video.mp4",
        "duration_seconds": 600.0,
        "fps": 30.0,
        "resolution": "1920x1080",
    }
    response = client.post("/api/videos", json=video_data)
    return response.json()


class TestStartDetection:
    """Tests for POST /videos/{video_id}/detect endpoint."""

    def test_start_detection_video_not_found(self):
        """Test starting detection for non-existent video."""
        response = client.post("/api/videos/999/detect")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_start_detection_success(self, sample_video):
        """Test successfully starting a detection job."""
        mock_job = MagicMock()
        mock_job.id = "test-job-id"

        with patch("app.api.detection.create_detection_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_job

            response = client.post(f"/api/videos/{sample_video['id']}/detect")

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["video_id"] == sample_video["id"]
        assert "message" in data

    def test_start_detection_with_custom_params(self, sample_video):
        """Test starting detection with custom parameters."""
        mock_job = MagicMock()
        mock_job.id = "test-job-id"

        with patch("app.api.detection.create_detection_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_job

            response = client.post(
                f"/api/videos/{sample_video['id']}/detect",
                json={
                    "sample_interval": 5,
                    "batch_size": 16,
                    "confidence_threshold": 0.7,
                },
            )

        assert response.status_code == 202

        # Verify the job was created with correct params
        call_args = mock_create.call_args
        params = call_args.kwargs["parameters"]
        assert params["sample_interval"] == 5
        assert params["batch_size"] == 16
        assert params["confidence_threshold"] == 0.7


class TestGetJobStatus:
    """Tests for GET /jobs/{job_id} endpoint."""

    def test_get_job_not_found(self):
        """Test getting status of non-existent job."""
        with patch("app.api.detection.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = client.get("/api/jobs/nonexistent-job-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_job_status_success(self):
        """Test getting job status successfully."""
        from app.models.processing_job import JobStatus, JobType

        mock_job = MagicMock()
        mock_job.id = "test-job-123"
        mock_job.job_type = JobType.VIDEO_DETECTION
        mock_job.status = JobStatus.PROCESSING
        mock_job.progress_current = 50
        mock_job.progress_total = 100
        mock_job.progress_percentage = 50.0
        mock_job.progress_message = "Processing frames..."
        mock_job.result = None
        mock_job.error_message = None
        mock_job.created_at = datetime.now()
        mock_job.started_at = datetime.now()
        mock_job.completed_at = None
        mock_job.parameters = {"video_id": 1}

        with patch("app.api.detection.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_job
            response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-job-123"
        assert data["job_type"] == "video_detection"
        assert data["status"] == "processing"
        assert data["progress"]["current"] == 50
        assert data["progress"]["total"] == 100
        assert data["progress"]["percentage"] == 50.0


class TestCancelJob:
    """Tests for DELETE /jobs/{job_id} endpoint."""

    def test_cancel_job_not_found(self):
        """Test cancelling non-existent job."""
        with patch("app.api.detection.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = client.delete("/api/jobs/nonexistent-job-id")

        assert response.status_code == 404

    def test_cancel_completed_job(self):
        """Test cancelling an already completed job."""
        from app.models.processing_job import JobStatus, JobType

        mock_job = MagicMock()
        mock_job.id = "completed-job"
        mock_job.job_type = JobType.VIDEO_DETECTION
        mock_job.status = JobStatus.COMPLETED

        with patch("app.api.detection.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_job
            response = client.delete("/api/jobs/completed-job")

        assert response.status_code == 400
        assert "cannot cancel" in response.json()["detail"].lower()

    def test_cancel_job_success(self):
        """Test successfully cancelling a job."""
        from app.models.processing_job import JobStatus, JobType

        mock_job = MagicMock()
        mock_job.id = "pending-job"
        mock_job.job_type = JobType.VIDEO_DETECTION
        mock_job.status = JobStatus.PENDING

        with patch("app.api.detection.get_job", new_callable=AsyncMock) as mock_get:
            with patch("app.api.detection.cancel_db_job", new_callable=AsyncMock) as mock_cancel:
                mock_get.return_value = mock_job
                mock_cancel.return_value = True

                response = client.delete("/api/jobs/pending-job")

        assert response.status_code == 204


class TestGetVideoDetections:
    """Tests for GET /videos/{video_id}/detections endpoint."""

    def test_get_detections_video_not_found(self):
        """Test getting detections for non-existent video."""
        response = client.get("/api/videos/999/detections")
        assert response.status_code == 404

    def test_get_detections_empty(self, sample_video):
        """Test getting detections when none exist."""
        response = client.get(f"/api/videos/{sample_video['id']}/detections")

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == sample_video["id"]
        assert data["total_detections"] == 0
        assert data["detections"] == []


class TestGetDetectionStats:
    """Tests for GET /videos/{video_id}/detections/stats endpoint."""

    def test_get_stats_video_not_found(self):
        """Test getting stats for non-existent video."""
        response = client.get("/api/videos/999/detections/stats")
        assert response.status_code == 404

    def test_get_stats_empty(self, sample_video):
        """Test getting stats when no detections exist."""
        response = client.get(f"/api/videos/{sample_video['id']}/detections/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == sample_video["id"]
        assert data["total_detections"] == 0
        assert data["frames_with_detections"] == 0


class TestDeleteVideoDetections:
    """Tests for DELETE /videos/{video_id}/detections endpoint."""

    def test_delete_detections_video_not_found(self):
        """Test deleting detections for non-existent video."""
        response = client.delete("/api/videos/999/detections")
        assert response.status_code == 404

    def test_delete_detections_success(self, sample_video):
        """Test successfully deleting all detections for a video (when empty)."""
        response = client.get(f"/api/videos/{sample_video['id']}/detections")
        assert response.json()["total_detections"] == 0

        response = client.delete(f"/api/videos/{sample_video['id']}/detections")
        assert response.status_code == 204

        response = client.get(f"/api/videos/{sample_video['id']}/detections")
        assert response.json()["total_detections"] == 0


class TestReprocessTracks:
    """Tests for POST /videos/{video_id}/reprocess-tracks endpoint."""

    def test_reprocess_tracks_video_not_found(self):
        response = client.post("/api/videos/999/reprocess-tracks")
        assert response.status_code == 404

    def test_reprocess_tracks_empty_video(self, sample_video):
        response = client.post(f"/api/videos/{sample_video['id']}/reprocess-tracks")

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == sample_video["id"]
        assert data["identity_switches_detected"] == 0
        assert data["tracks_before_merge"] == 0
        assert data["tracks_after_merge"] == 0

    def test_reprocess_tracks_with_options(self, sample_video):
        response = client.post(
            f"/api/videos/{sample_video['id']}/reprocess-tracks",
            json={
                "enable_identity_switch_detection": True,
                "enable_track_merging": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == sample_video["id"]
        assert data["tracks_before_merge"] == 0
        assert data["tracks_after_merge"] == 0
