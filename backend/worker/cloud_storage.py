"""R2 cloud storage operations for cloud GPU processing."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


@dataclass
class JobManifest:
    """Job manifest stored in R2."""

    job_id: str
    video_id: int
    status: str  # pending, processing, completed, failed, imported
    created_at: str
    parameters: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    frames_processed: int = 0
    worker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status,
            "created_at": self.created_at,
            "parameters": self.parameters,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "frames_processed": self.frames_processed,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        return cls(
            job_id=data["job_id"],
            video_id=data["video_id"],
            status=data["status"],
            created_at=data["created_at"],
            parameters=data.get("parameters", {}),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            frames_processed=data.get("frames_processed", 0),
            worker_id=data.get("worker_id"),
        )


class CloudStorage:
    """R2 cloud storage client for job queue operations."""

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
    ) -> None:
        self._bucket = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        logger.info(f"CloudStorage initialized for bucket: {bucket_name}")

    def upload_video(self, job_id: str, video_path: Path) -> str:
        """Upload video file to R2.

        Args:
            job_id: Job ID for naming.
            video_path: Local path to video file.

        Returns:
            R2 key for the uploaded video.

        Raises:
            Exception: If upload fails.
        """
        key = f"videos/{job_id}{video_path.suffix}"
        logger.info(f"Uploading video to {key}...")
        try:
            self._client.upload_file(str(video_path), self._bucket, key)
            logger.info(f"Uploaded video: {key}")
            return key
        except Exception as e:
            logger.error(f"Failed to upload video {video_path} to {key}: {e}")
            raise

    def download_video(self, job_id: str, dest_path: Path, suffix: str = ".mp4") -> None:
        """Download video file from R2.

        Args:
            job_id: Job ID.
            dest_path: Local destination path.
            suffix: Video file suffix.

        Raises:
            Exception: If download fails.
        """
        key = f"videos/{job_id}{suffix}"
        logger.info(f"Downloading video from {key}...")
        try:
            self._client.download_file(self._bucket, key, str(dest_path))
            logger.info(f"Downloaded video to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to download video from {key} to {dest_path}: {e}")
            raise

    def upload_job_manifest(self, manifest: JobManifest) -> None:
        """Upload job manifest to R2.

        Raises:
            Exception: If upload fails.
        """
        key = f"jobs/{manifest.job_id}.json"
        body = json.dumps(manifest.to_dict(), indent=2)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.debug(f"Uploaded job manifest: {key}")
        except Exception as e:
            logger.error(f"Failed to upload job manifest for {manifest.job_id}: {e}")
            raise

    def get_job_manifest(self, job_id: str) -> JobManifest | None:
        """Get job manifest from R2."""
        key = f"jobs/{job_id}.json"
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return JobManifest.from_dict(data)
        except self._client.exceptions.NoSuchKey:
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode job manifest JSON for {job_id}: {e}")
            return None

    def list_pending_jobs(self) -> list[JobManifest]:
        """List all pending jobs from R2."""
        jobs = []
        response = self._client.list_objects_v2(Bucket=self._bucket, Prefix="jobs/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                manifest_response = self._client.get_object(Bucket=self._bucket, Key=key)
                data = json.loads(manifest_response["Body"].read().decode("utf-8"))
                manifest = JobManifest.from_dict(data)
                if manifest.status == "pending":
                    jobs.append(manifest)
            except Exception as e:
                logger.warning(f"Failed to read job manifest {key}: {e}")
        return sorted(jobs, key=lambda j: j.created_at)

    def upload_results(self, job_id: str, results: dict[str, Any]) -> None:
        """Upload detection results to R2.

        Raises:
            Exception: If upload fails.
        """
        key = f"results/{job_id}.json"
        body = json.dumps(results)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"Uploaded results: {key}")
        except Exception as e:
            logger.error(f"Failed to upload results for {job_id}: {e}")
            raise

    def download_results(self, job_id: str) -> dict[str, Any] | None:
        """Download detection results from R2."""
        key = f"results/{job_id}.json"
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode results JSON for {job_id}: {e}")
            return None

    def update_status(self, job_id: str, current: int, total: int, message: str = "") -> None:
        """Update job status file for progress tracking."""
        key = f"status/{job_id}.json"
        body = json.dumps({
            "job_id": job_id,
            "current": current,
            "total": total,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status from R2."""
        key = f"status/{job_id}.json"
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode status JSON for {job_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get status for {job_id}: {e}")
            return None

    def delete_job_files(self, job_id: str) -> None:
        """Delete all files for a job (cleanup after import)."""
        prefixes = [f"videos/{job_id}", f"jobs/{job_id}", f"results/{job_id}", f"status/{job_id}"]
        deleted_count = 0
        for prefix in prefixes:
            response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            for obj in response.get("Contents", []):
                self._client.delete_object(Bucket=self._bucket, Key=obj["Key"])
                logger.debug(f"Deleted: {obj['Key']}")
                deleted_count += 1
        logger.info(f"Cleanup complete for job {job_id}: deleted {deleted_count} files")
