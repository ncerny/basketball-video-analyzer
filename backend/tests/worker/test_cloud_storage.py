"""Tests for R2 cloud storage operations."""

import json
import pytest
from unittest.mock import MagicMock, patch

from worker.cloud_storage import CloudStorage, JobManifest


class TestJobManifest:
    """Tests for JobManifest dataclass."""

    def test_to_dict(self):
        manifest = JobManifest(
            job_id="test-123",
            video_id=1,
            status="pending",
            created_at="2026-01-22T10:00:00Z",
            parameters={"sample_interval": 1},
        )
        result = manifest.to_dict()
        assert result["job_id"] == "test-123"
        assert result["video_id"] == 1
        assert result["status"] == "pending"

    def test_from_dict(self):
        data = {
            "job_id": "test-123",
            "video_id": 1,
            "status": "pending",
            "created_at": "2026-01-22T10:00:00Z",
            "parameters": {"sample_interval": 1},
        }
        manifest = JobManifest.from_dict(data)
        assert manifest.job_id == "test-123"
        assert manifest.video_id == 1


class TestCloudStorage:
    """Tests for CloudStorage R2 operations."""

    @pytest.fixture
    def mock_s3_client(self):
        with patch("worker.cloud_storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            yield mock_client

    def test_init_creates_client(self, mock_s3_client):
        storage = CloudStorage(
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
            bucket_name="test-bucket",
        )
        assert storage._bucket == "test-bucket"

    def test_upload_job_manifest(self, mock_s3_client):
        storage = CloudStorage(
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
            bucket_name="test-bucket",
        )
        manifest = JobManifest(
            job_id="test-123",
            video_id=1,
            status="pending",
            created_at="2026-01-22T10:00:00Z",
            parameters={},
        )
        storage.upload_job_manifest(manifest)
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args.kwargs["Bucket"] == "test-bucket"
        assert call_args.kwargs["Key"] == "jobs/test-123.json"

    def test_list_pending_jobs(self, mock_s3_client):
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "jobs/job-1.json"}, {"Key": "jobs/job-2.json"}]
        }
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(
                read=lambda: json.dumps({
                    "job_id": "job-1",
                    "video_id": 1,
                    "status": "pending",
                    "created_at": "2026-01-22T10:00:00Z",
                    "parameters": {},
                }).encode()
            )
        }
        storage = CloudStorage(
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
            bucket_name="test-bucket",
        )
        jobs = storage.list_pending_jobs()
        assert len(jobs) >= 1
