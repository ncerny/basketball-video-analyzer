"""Tests for detection API endpoints."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.job_manager import Job, JobProgress, JobStatus, reset_job_manager

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


@pytest.fixture(autouse=True)
def reset_job_manager_fixture():
    """Reset job manager before and after each test."""
    reset_job_manager()
    yield
    reset_job_manager()


class TestStartDetection:
    """Tests for POST /videos/{video_id}/detect endpoint."""

    def test_start_detection_video_not_found(self):
        """Test starting detection for non-existent video."""
        response = client.post("/api/videos/999/detect")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_start_detection_success(self, sample_video):
        """Test successfully starting a detection job."""
        with patch("app.api.detection._ensure_detection_worker_registered", new_callable=AsyncMock):
            with patch("app.api.detection.get_job_manager") as mock_get_jm:
                mock_jm = MagicMock()
                mock_jm.submit_job = AsyncMock(return_value="test-job-id")
                mock_get_jm.return_value = mock_jm

                response = client.post(f"/api/videos/{sample_video['id']}/detect")

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["video_id"] == sample_video["id"]
        assert "message" in data

    def test_start_detection_with_custom_params(self, sample_video):
        """Test starting detection with custom parameters."""
        with patch("app.api.detection._ensure_detection_worker_registered", new_callable=AsyncMock):
            with patch("app.api.detection.get_job_manager") as mock_get_jm:
                mock_jm = MagicMock()
                mock_jm.submit_job = AsyncMock(return_value="test-job-id")
                mock_get_jm.return_value = mock_jm

                response = client.post(
                    f"/api/videos/{sample_video['id']}/detect",
                    json={
                        "sample_interval": 5,
                        "batch_size": 16,
                        "confidence_threshold": 0.7,
                    },
                )

        assert response.status_code == 202

        # Verify the job was submitted with correct params
        call_args = mock_jm.submit_job.call_args
        assert call_args.kwargs["metadata"]["sample_interval"] == 5
        assert call_args.kwargs["metadata"]["batch_size"] == 16
        assert call_args.kwargs["metadata"]["confidence_threshold"] == 0.7


class TestGetJobStatus:
    """Tests for GET /jobs/{job_id} endpoint."""

    def test_get_job_not_found(self):
        """Test getting status of non-existent job."""
        response = client.get("/api/jobs/nonexistent-job-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_job_status_success(self):
        """Test getting job status successfully."""
        # Create a mock job
        job = Job(
            id="test-job-123",
            job_type="video_detection",
            status=JobStatus.PROCESSING,
            progress=JobProgress(current=50, total=100, message="Processing frames..."),
            metadata={"video_id": 1},
        )

        with patch("app.api.detection.get_job_manager") as mock_get_jm:
            mock_jm = MagicMock()
            mock_jm.get_job.return_value = job
            mock_get_jm.return_value = mock_jm

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
        response = client.delete("/api/jobs/nonexistent-job-id")
        assert response.status_code == 404

    def test_cancel_completed_job(self):
        """Test cancelling an already completed job."""
        job = Job(
            id="completed-job",
            job_type="video_detection",
            status=JobStatus.COMPLETED,
        )

        with patch("app.api.detection.get_job_manager") as mock_get_jm:
            mock_jm = MagicMock()
            mock_jm.get_job.return_value = job
            mock_get_jm.return_value = mock_jm

            response = client.delete("/api/jobs/completed-job")

        assert response.status_code == 400
        assert "cannot cancel" in response.json()["detail"].lower()

    def test_cancel_job_success(self):
        """Test successfully cancelling a job."""
        job = Job(
            id="pending-job",
            job_type="video_detection",
            status=JobStatus.PENDING,
        )

        with patch("app.api.detection.get_job_manager") as mock_get_jm:
            mock_jm = MagicMock()
            mock_jm.get_job.return_value = job
            mock_jm.cancel_job = AsyncMock(return_value=True)
            mock_get_jm.return_value = mock_jm

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
        # First verify we can get detections
        response = client.get(f"/api/videos/{sample_video['id']}/detections")
        assert response.json()["total_detections"] == 0

        # Delete detections (even when empty, should succeed)
        response = client.delete(f"/api/videos/{sample_video['id']}/detections")
        assert response.status_code == 204

        # Verify still empty
        response = client.get(f"/api/videos/{sample_video['id']}/detections")
        assert response.json()["total_detections"] == 0
