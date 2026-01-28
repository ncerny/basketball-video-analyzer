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
    last_heartbeat: str | None = None  # Updated on each progress update

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
            "last_heartbeat": self.last_heartbeat,
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
            last_heartbeat=data.get("last_heartbeat"),
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
        self.download_video_by_key(key, dest_path)

    def download_video_by_key(self, key: str, dest_path: Path) -> None:
        """Download video file from R2 by full key.

        Args:
            key: Full R2 key (e.g., 'videos/job-id.mov').
            dest_path: Local destination path.

        Raises:
            Exception: If download fails.
        """
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

    def reset_orphaned_jobs(self, stale_minutes: int = 10) -> list[str]:
        """Reset orphaned 'processing' jobs back to 'pending'.

        A job is considered orphaned if it's been processing for longer than
        stale_minutes without a heartbeat update.

        Args:
            stale_minutes: Minutes without heartbeat before job is orphaned.

        Returns:
            List of job IDs that were reset.
        """
        from datetime import datetime, timezone

        reset_jobs = []
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (stale_minutes * 60)

        response = self._client.list_objects_v2(Bucket=self._bucket, Prefix="jobs/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                manifest_response = self._client.get_object(Bucket=self._bucket, Key=key)
                data = json.loads(manifest_response["Body"].read().decode("utf-8"))
                manifest = JobManifest.from_dict(data)

                if manifest.status != "processing":
                    continue

                # Check heartbeat (fall back to started_at if no heartbeat)
                check_time = manifest.last_heartbeat or manifest.started_at
                if check_time:
                    try:
                        job_time = datetime.fromisoformat(check_time.replace("Z", "+00:00"))
                        if job_time.timestamp() > cutoff:
                            continue  # Not stale yet
                    except ValueError:
                        pass  # Invalid timestamp, consider it stale

                # Reset to pending
                logger.warning(
                    f"Resetting orphaned job {manifest.job_id} "
                    f"(was processing since {manifest.started_at}, "
                    f"last heartbeat: {manifest.last_heartbeat})"
                )
                manifest.status = "pending"
                manifest.started_at = None
                manifest.last_heartbeat = None
                manifest.worker_id = None
                self.upload_job_manifest(manifest)
                reset_jobs.append(manifest.job_id)

            except Exception as e:
                logger.warning(f"Failed to check job manifest {key}: {e}")

        return reset_jobs

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

    def find_jobs_for_video(self, video_id: int) -> list[JobManifest]:
        """Find all jobs for a given video ID.

        Args:
            video_id: Database video ID.

        Returns:
            List of job manifests for this video, sorted by created_at desc.
        """
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
                if manifest.video_id == video_id:
                    jobs.append(manifest)
            except Exception as e:
                logger.warning(f"Failed to read job manifest {key}: {e}")
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def video_exists(self, key: str) -> bool:
        """Check if a video exists in R2.

        Args:
            key: R2 key for the video.

        Returns:
            True if video exists, False otherwise.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def delete_job_files(self, job_id: str, keep_video: bool = True) -> None:
        """Delete job files (cleanup after import).

        Args:
            job_id: Job ID to clean up.
            keep_video: If True, preserves video file for streaming. Default True.
        """
        prefixes = [f"jobs/{job_id}", f"results/{job_id}", f"status/{job_id}"]
        if not keep_video:
            prefixes.insert(0, f"videos/{job_id}")
        deleted_count = 0
        for prefix in prefixes:
            response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            for obj in response.get("Contents", []):
                self._client.delete_object(Bucket=self._bucket, Key=obj["Key"])
                logger.debug(f"Deleted: {obj['Key']}")
                deleted_count += 1
        logger.info(f"Cleanup complete for job {job_id}: deleted {deleted_count} files")

    def _get_gpu_cache_key(self) -> str | None:
        """Get cache key based on GPU model for torch compile cache.

        Returns:
            Cache key like 'cache/torch-compile-rtx-pro-6000.tar.gz' or None if no CUDA GPU.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            gpu_name = torch.cuda.get_device_name(0)
            # Sanitize GPU name for use in filename
            safe_name = gpu_name.lower().replace(" ", "-").replace("/", "-")
            # Remove common prefixes
            for prefix in ["nvidia-", "geforce-"]:
                if safe_name.startswith(prefix):
                    safe_name = safe_name[len(prefix):]
            return f"cache/torch-compile-{safe_name}.tar.gz"
        except Exception:
            return None

    def download_torch_cache(self, cache_dir: Path | None = None) -> bool:
        """Download torch compile cache from R2 if it exists.

        Args:
            cache_dir: Local cache directory. Defaults to ~/.cache/torch.

        Returns:
            True if cache was downloaded, False if not found or error.
        """
        import os
        import tarfile
        import tempfile

        cache_key = self._get_gpu_cache_key()
        if not cache_key:
            logger.debug("No CUDA GPU detected, skipping torch cache download")
            return False

        cache_dir = cache_dir or Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))

        try:
            # Check if cache exists in R2
            try:
                self._client.head_object(Bucket=self._bucket, Key=cache_key)
            except self._client.exceptions.ClientError:
                logger.debug("No torch compile cache found in R2")
                return False

            # Download to temp file and extract
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name

            logger.info(f"Downloading torch compile cache from R2: {cache_key}")
            self._client.download_file(self._bucket, cache_key, tmp_path)

            # Extract to cache directory
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(cache_dir)

            Path(tmp_path).unlink()
            logger.info(f"Restored torch compile cache to {cache_dir}")
            return True

        except Exception as e:
            logger.warning(f"Failed to download torch cache: {e}")
            return False

    def upload_torch_cache(self, cache_dir: Path | None = None) -> bool:
        """Upload torch compile cache to R2.

        Args:
            cache_dir: Local cache directory. Defaults to ~/.cache/torch.

        Returns:
            True if cache was uploaded, False if error or nothing to upload.
        """
        import os
        import tarfile
        import tempfile

        cache_key = self._get_gpu_cache_key()
        if not cache_key:
            logger.debug("No CUDA GPU detected, skipping torch cache upload")
            return False

        cache_dir = cache_dir or Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))

        # Check for inductor cache (where compiled kernels are stored)
        inductor_cache = cache_dir / "inductor"
        if not inductor_cache.exists():
            logger.debug("No torch inductor cache to upload")
            return False

        try:
            # Create tarball of cache
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name

            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(inductor_cache, arcname="inductor")

            # Upload to R2
            file_size = Path(tmp_path).stat().st_size / 1024 / 1024
            logger.info(f"Uploading torch compile cache ({file_size:.1f} MB) to R2: {cache_key}")
            self._client.upload_file(tmp_path, self._bucket, cache_key)

            Path(tmp_path).unlink()
            logger.info("Torch compile cache uploaded to R2")
            return True

        except Exception as e:
            logger.warning(f"Failed to upload torch cache: {e}")
            return False
