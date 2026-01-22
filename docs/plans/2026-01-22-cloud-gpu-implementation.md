# Cloud GPU Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable SAM3 video detection to run on cloud GPU via R2 file-based job queue.

**Architecture:** Local CLI submits jobs to R2 (video + manifest), cloud worker polls R2, processes, uploads results. Local CLI imports results into SQLite.

**Tech Stack:** boto3 (R2/S3 API), click (CLI), Docker (containerization)

**Design Doc:** `docs/plans/2026-01-22-cloud-gpu-processing-design.md`

---

## Task 1: Add Dependencies

**Files:**
- Modify: `backend/pyproject.toml`

**Step 1: Add boto3 and click to dependencies**

In `pyproject.toml`, add to `[tool.poetry.dependencies]`:

```toml
boto3 = "^1.34.0"
click = "^8.1.0"
```

**Step 2: Install dependencies**

Run: `cd backend && poetry install`

**Step 3: Commit**

```bash
git add backend/pyproject.toml backend/poetry.lock
git commit -m "chore: add boto3 and click dependencies for cloud worker"
```

---

## Task 2: Add R2 Configuration Settings

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/.env` (local only, not committed)

**Step 1: Add R2 settings to config.py**

Add after the existing `use_external_worker` setting:

```python
    # Cloud storage (Cloudflare R2)
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "basketball-analyzer"

    @property
    def r2_endpoint_url(self) -> str:
        """Generate R2 endpoint URL from account ID."""
        if not self.r2_account_id:
            return ""
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    # Cloud worker settings
    cloud_worker_poll_interval: float = 10.0
    cloud_model_path: str = "/models/sam3"
```

**Step 2: Add to local .env (do not commit)**

```bash
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-key
R2_SECRET_ACCESS_KEY=your-secret
R2_BUCKET_NAME=basketball-analyzer
```

**Step 3: Commit config.py only**

```bash
git add backend/app/config.py
git commit -m "feat(config): add R2 cloud storage settings"
```

---

## Task 3: Create Cloud Storage Module

**Files:**
- Create: `backend/worker/cloud_storage.py`
- Create: `backend/tests/worker/test_cloud_storage.py`

**Step 1: Create test file**

```python
# backend/tests/worker/test_cloud_storage.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/worker/test_cloud_storage.py -v`
Expected: FAIL (module not found)

**Step 3: Create cloud_storage.py**

```python
# backend/worker/cloud_storage.py
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
        """
        key = f"videos/{job_id}{video_path.suffix}"
        logger.info(f"Uploading video to {key}...")
        self._client.upload_file(str(video_path), self._bucket, key)
        logger.info(f"Uploaded video: {key}")
        return key

    def download_video(self, job_id: str, dest_path: Path, suffix: str = ".mp4") -> None:
        """Download video file from R2.

        Args:
            job_id: Job ID.
            dest_path: Local destination path.
            suffix: Video file suffix.
        """
        key = f"videos/{job_id}{suffix}"
        logger.info(f"Downloading video from {key}...")
        self._client.download_file(self._bucket, key, str(dest_path))
        logger.info(f"Downloaded video to: {dest_path}")

    def upload_job_manifest(self, manifest: JobManifest) -> None:
        """Upload job manifest to R2."""
        key = f"jobs/{manifest.job_id}.json"
        body = json.dumps(manifest.to_dict(), indent=2)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.debug(f"Uploaded job manifest: {key}")

    def get_job_manifest(self, job_id: str) -> JobManifest | None:
        """Get job manifest from R2."""
        key = f"jobs/{job_id}.json"
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return JobManifest.from_dict(data)
        except self._client.exceptions.NoSuchKey:
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
        """Upload detection results to R2."""
        key = f"results/{job_id}.json"
        body = json.dumps(results)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Uploaded results: {key}")

    def download_results(self, job_id: str) -> dict[str, Any] | None:
        """Download detection results from R2."""
        key = f"results/{job_id}.json"
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
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
        except:
            return None

    def delete_job_files(self, job_id: str) -> None:
        """Delete all files for a job (cleanup after import)."""
        prefixes = [f"videos/{job_id}", f"jobs/{job_id}", f"results/{job_id}", f"status/{job_id}"]
        for prefix in prefixes:
            response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            for obj in response.get("Contents", []):
                self._client.delete_object(Bucket=self._bucket, Key=obj["Key"])
                logger.debug(f"Deleted: {obj['Key']}")
```

**Step 4: Create tests directory if needed**

Run: `mkdir -p backend/tests/worker && touch backend/tests/worker/__init__.py`

**Step 5: Run tests**

Run: `cd backend && pytest tests/worker/test_cloud_storage.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backend/worker/cloud_storage.py backend/tests/worker/
git commit -m "feat(worker): add R2 cloud storage module"
```

---

## Task 4: Create CLI Module

**Files:**
- Create: `backend/worker/cli.py`
- Modify: `backend/worker/__init__.py`

**Step 1: Create cli.py**

```python
# backend/worker/cli.py
"""CLI for cloud job management: submit, import, status."""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from worker.cloud_storage import CloudStorage, JobManifest


def get_storage() -> CloudStorage:
    """Get configured CloudStorage instance."""
    if not settings.r2_account_id:
        click.echo("Error: R2 not configured. Set R2_ACCOUNT_ID in .env", err=True)
        raise SystemExit(1)
    return CloudStorage(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
    )


@click.group()
def cli():
    """Cloud GPU job management for basketball video analysis."""
    pass


@cli.command()
@click.option("--video-id", required=True, type=int, help="Database video ID")
@click.option("--video-path", required=True, type=click.Path(exists=True), help="Path to video file")
@click.option("--sample-interval", default=1, type=int, help="Process every Nth frame")
@click.option("--confidence", default=0.25, type=float, help="Confidence threshold")
def submit(video_id: int, video_path: str, sample_interval: int, confidence: float):
    """Submit a video for cloud GPU processing."""
    storage = get_storage()
    video_path = Path(video_path)

    # Generate job ID
    job_id = str(uuid.uuid4())
    click.echo(f"Creating job {job_id}...")

    # Upload video
    click.echo(f"Uploading video ({video_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    video_key = storage.upload_video(job_id, video_path)

    # Create and upload manifest
    manifest = JobManifest(
        job_id=job_id,
        video_id=video_id,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
        parameters={
            "sample_interval": sample_interval,
            "confidence_threshold": confidence,
            "video_key": video_key,
        },
    )
    storage.upload_job_manifest(manifest)

    click.echo(f"Job submitted: {job_id}")
    click.echo(f"Start cloud worker to process, then run: python -m worker.cli import-job --job-id {job_id}")


@cli.command("status")
def status():
    """Show status of all cloud jobs."""
    storage = get_storage()

    # Get all job manifests
    response = storage._client.list_objects_v2(Bucket=storage._bucket, Prefix="jobs/")
    jobs = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        manifest = storage.get_job_manifest(key.split("/")[1].replace(".json", ""))
        if manifest:
            jobs.append(manifest)

    if not jobs:
        click.echo("No cloud jobs found.")
        return

    # Sort by created_at
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Print table
    click.echo(f"{'JOB_ID':<38} {'VIDEO':<6} {'STATUS':<12} {'PROGRESS':<20}")
    click.echo("-" * 80)
    for job in jobs:
        # Get progress if processing
        progress = ""
        if job.status == "processing":
            status_data = storage.get_status(job.job_id)
            if status_data:
                progress = f"{status_data.get('current', 0)}/{status_data.get('total', '?')} frames"
        elif job.status == "completed":
            progress = "ready to import"
        elif job.status == "imported":
            progress = "imported"
        elif job.status == "failed":
            progress = job.error or "error"

        click.echo(f"{job.job_id:<38} {job.video_id:<6} {job.status:<12} {progress:<20}")


@cli.command("import-job")
@click.option("--job-id", required=True, help="Job ID to import")
@click.option("--cleanup/--no-cleanup", default=True, help="Delete R2 files after import")
def import_job(job_id: str, cleanup: bool):
    """Import completed job results into local database."""
    import asyncio
    from sqlalchemy import select
    from app.database import async_session_maker
    from app.models.detection import PlayerDetection
    from app.models.video import Video

    storage = get_storage()

    # Check job status
    manifest = storage.get_job_manifest(job_id)
    if not manifest:
        click.echo(f"Job {job_id} not found", err=True)
        raise SystemExit(1)

    if manifest.status != "completed":
        click.echo(f"Job {job_id} is not completed (status: {manifest.status})", err=True)
        raise SystemExit(1)

    # Download results
    click.echo(f"Downloading results for job {job_id}...")
    results = storage.download_results(job_id)
    if not results:
        click.echo(f"No results found for job {job_id}", err=True)
        raise SystemExit(1)

    detections = results.get("detections", [])
    click.echo(f"Found {len(detections)} detections")

    # Import into database
    async def do_import():
        async with async_session_maker() as session:
            # Verify video exists
            video_result = await session.execute(
                select(Video).where(Video.id == manifest.video_id)
            )
            video = video_result.scalar_one_or_none()
            if not video:
                click.echo(f"Video {manifest.video_id} not found in database", err=True)
                raise SystemExit(1)

            # Insert detections
            for det in detections:
                detection = PlayerDetection(
                    video_id=manifest.video_id,
                    frame_number=det["frame"],
                    tracking_id=det["track_id"],
                    bbox_x=det["bbox"][0],
                    bbox_y=det["bbox"][1],
                    bbox_width=det["bbox"][2],
                    bbox_height=det["bbox"][3],
                    confidence_score=det["confidence"],
                )
                session.add(detection)

            await session.commit()
            click.echo(f"Imported {len(detections)} detections into database")

    asyncio.run(do_import())

    # Update manifest status
    manifest.status = "imported"
    storage.upload_job_manifest(manifest)

    # Cleanup R2 files
    if cleanup:
        click.echo("Cleaning up R2 files...")
        storage.delete_job_files(job_id)
        click.echo("Cleanup complete")


@cli.command("import-all")
@click.option("--cleanup/--no-cleanup", default=True, help="Delete R2 files after import")
def import_all(cleanup: bool):
    """Import all completed jobs."""
    storage = get_storage()

    # Find completed jobs
    response = storage._client.list_objects_v2(Bucket=storage._bucket, Prefix="jobs/")
    completed = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        job_id = key.split("/")[1].replace(".json", "")
        manifest = storage.get_job_manifest(job_id)
        if manifest and manifest.status == "completed":
            completed.append(manifest)

    if not completed:
        click.echo("No completed jobs to import.")
        return

    click.echo(f"Found {len(completed)} completed jobs to import")
    for manifest in completed:
        click.echo(f"\nImporting {manifest.job_id}...")
        # Use click's invoke to call import_job
        ctx = click.Context(import_job)
        ctx.invoke(import_job, job_id=manifest.job_id, cleanup=cleanup)


if __name__ == "__main__":
    cli()
```

**Step 2: Update worker __init__.py**

```python
# backend/worker/__init__.py
"""Detection worker module.

Run as a separate process to handle ML processing jobs (SAM3 detection, etc.)
independently from the FastAPI backend.

Usage (local):
    python -m worker

Usage (cloud CLI):
    python -m worker.cli submit --video-id 1 --video-path /path/to/video.mp4
    python -m worker.cli status
    python -m worker.cli import-all

Or use the convenience script:
    ./scripts/start_worker.sh
"""

from worker.config import WorkerConfig
from worker.job_processor import JobProcessor

__all__ = ["JobProcessor", "WorkerConfig"]
```

**Step 3: Test CLI runs**

Run: `cd backend && python -m worker.cli --help`
Expected: Shows help with submit, status, import-job, import-all commands

**Step 4: Commit**

```bash
git add backend/worker/cli.py backend/worker/__init__.py
git commit -m "feat(worker): add CLI for cloud job management"
```

---

## Task 5: Create Cloud Worker

**Files:**
- Create: `backend/worker/cloud_worker.py`

**Step 1: Create cloud_worker.py**

```python
# backend/worker/cloud_worker.py
"""Cloud worker that polls R2 for jobs and processes them."""

import asyncio
import logging
import signal
import socket
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from worker.cloud_storage import CloudStorage, JobManifest
from worker.config import WorkerConfig

logger = logging.getLogger(__name__)


class CloudWorker:
    """Worker that polls R2 for pending jobs and processes them."""

    def __init__(self, storage: CloudStorage, config: WorkerConfig) -> None:
        self._storage = storage
        self._config = config
        self._worker_id = config.worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    async def run(self, shutdown_event: asyncio.Event, single_job: bool = False) -> None:
        """Main processing loop.

        Args:
            shutdown_event: Event to signal graceful shutdown.
            single_job: If True, exit after processing one job.
        """
        logger.info(f"Cloud worker {self._worker_id} starting...")

        while not shutdown_event.is_set():
            # Poll for pending jobs
            jobs = self._storage.list_pending_jobs()

            if not jobs:
                if single_job:
                    logger.info("No pending jobs found, exiting")
                    break
                logger.debug("No pending jobs, waiting...")
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=self._config.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            # Process first pending job
            job = jobs[0]
            logger.info(f"Claiming job {job.job_id} (video_id={job.video_id})")

            try:
                await self._process_job(job)
            except Exception as e:
                logger.exception(f"Job {job.job_id} failed: {e}")
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc).isoformat()
                self._storage.upload_job_manifest(job)

            if single_job:
                break

        logger.info("Cloud worker stopped")

    async def _process_job(self, job: JobManifest) -> None:
        """Process a single job."""
        # Mark as processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.worker_id = self._worker_id
        self._storage.upload_job_manifest(job)

        # Download video to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / f"{job.job_id}.mp4"
            self._storage.download_video(job.job_id, video_path)

            # Run detection
            detections = await self._run_detection(job, video_path)

            # Upload results
            results = {
                "job_id": job.job_id,
                "video_id": job.video_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "frames_processed": job.frames_processed,
                "detections": detections,
            }
            self._storage.upload_results(job.job_id, results)

        # Mark as completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._storage.upload_job_manifest(job)

        logger.info(f"Job {job.job_id} completed: {len(detections)} detections")

    async def _run_detection(self, job: JobManifest, video_path: Path) -> list[dict]:
        """Run SAM3 detection on video.

        Args:
            job: Job manifest with parameters.
            video_path: Path to downloaded video.

        Returns:
            List of detection dicts.
        """
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        params = job.parameters
        sample_interval = params.get("sample_interval", 1)
        confidence = params.get("confidence_threshold", 0.25)

        logger.info(f"Starting SAM3 detection (sample_interval={sample_interval})")

        pipeline = SAM3DetectionPipeline(confidence_threshold=confidence)
        detections = []
        frame_count = 0

        async for frame_detections in pipeline.process_video(
            video_path, sample_interval=sample_interval
        ):
            for det in frame_detections.detections:
                detections.append({
                    "frame": frame_detections.frame_number,
                    "track_id": det.tracking_id,
                    "bbox": [det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height],
                    "confidence": det.confidence,
                })

            frame_count += 1
            job.frames_processed = frame_count

            # Update progress every 50 frames
            if frame_count % 50 == 0:
                self._storage.update_status(
                    job.job_id,
                    current=frame_count,
                    total=-1,  # Unknown total
                    message=f"Processing frame {frame_count}",
                )
                # Also update manifest
                self._storage.upload_job_manifest(job)

        return detections


async def main():
    """Entry point for cloud worker."""
    import os
    from app.config import settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not settings.r2_account_id:
        logger.error("R2 not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(1)

    storage = CloudStorage(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
    )

    config = WorkerConfig(
        poll_interval_seconds=settings.cloud_worker_poll_interval,
        worker_id=os.environ.get("WORKER_ID", ""),
    )

    worker = CloudWorker(storage, config)

    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for single-job mode
    single_job = os.environ.get("SINGLE_JOB", "").lower() == "true"

    await worker.run(shutdown_event, single_job=single_job)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Test cloud worker imports**

Run: `cd backend && python -c "from worker.cloud_worker import CloudWorker; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add backend/worker/cloud_worker.py
git commit -m "feat(worker): add cloud worker that polls R2 for jobs"
```

---

## Task 6: Update Worker Entry Point for Cloud Mode

**Files:**
- Modify: `backend/worker/__main__.py`

**Step 1: Add --cloud flag to worker entry point**

Replace the content of `backend/worker/__main__.py`:

```python
# backend/worker/__main__.py
"""Entry point for running the worker as a module.

Usage:
    # Local worker (polls SQLite DB)
    python -m worker

    # Cloud worker (polls R2)
    python -m worker --cloud

    # CLI commands
    python -m worker.cli submit --video-id 1 --video-path video.mp4
    python -m worker.cli status
    python -m worker.cli import-all
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging() -> None:
    """Configure logging for the worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def run_local_worker() -> None:
    """Run local worker that polls SQLite DB."""
    from worker.config import WorkerConfig
    from worker.job_processor import JobProcessor

    logger = logging.getLogger("worker")
    config = WorkerConfig.from_env()

    logger.info(f"Starting LOCAL worker {config.worker_id}")
    logger.info(f"Database: {config.database_url}")

    processor = JobProcessor(config)
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await processor.run(shutdown_event)
    logger.info("Local worker stopped")


async def run_cloud_worker() -> None:
    """Run cloud worker that polls R2."""
    from worker.cloud_worker import main as cloud_main
    await cloud_main()


def main() -> None:
    """Main entry point."""
    setup_logging()

    # Check for --cloud flag
    cloud_mode = "--cloud" in sys.argv

    if cloud_mode:
        asyncio.run(run_cloud_worker())
    else:
        asyncio.run(run_local_worker())


if __name__ == "__main__":
    main()
```

**Step 2: Test both modes**

Run: `cd backend && python -m worker --help 2>&1 | head -5`
(Should start local worker, Ctrl+C to stop)

Run: `cd backend && timeout 2 python -m worker --cloud 2>&1 || true`
(Should fail with R2 not configured, which is expected)

**Step 3: Commit**

```bash
git add backend/worker/__main__.py
git commit -m "feat(worker): add --cloud flag for R2-based processing"
```

---

## Task 7: Create Dockerfile

**Files:**
- Create: `backend/Dockerfile.worker`

**Step 1: Create Dockerfile.worker**

```dockerfile
# backend/Dockerfile.worker
# Cloud worker image with SAM3 model baked in
#
# Build locally (with model downloaded):
#   docker build -f Dockerfile.worker -t basketball-analyzer-worker .
#
# Run:
#   docker run -e R2_ACCOUNT_ID=xxx -e R2_ACCESS_KEY_ID=xxx \
#     -e R2_SECRET_ACCESS_KEY=xxx -e R2_BUCKET_NAME=xxx \
#     basketball-analyzer-worker

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better caching)
COPY pyproject.toml poetry.lock* ./

# Install poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main,ml,video

# Install additional cloud dependencies
RUN pip install boto3 click

# Copy application code
COPY app/ ./app/
COPY worker/ ./worker/

# Copy pre-downloaded SAM3 model (build locally with model present)
# The model should be at models/sam3/ relative to the build context
COPY models/sam3 /models/sam3

# Set environment variables
ENV PYTHONPATH=/app
ENV CLOUD_MODEL_PATH=/models/sam3

# Default: run cloud worker
CMD ["python", "-m", "worker", "--cloud"]
```

**Step 2: Create .dockerignore**

```bash
cat > backend/.dockerignore << 'EOF'
# Ignore everything except what we need
*
!pyproject.toml
!poetry.lock
!app/
!worker/
!models/sam3/

# Ignore Python cache
**/__pycache__
**/*.pyc
**/*.pyo

# Ignore test files
**/tests/
**/*_test.py
**/test_*.py

# Ignore development files
.env
.venv/
*.db
*.db-*
EOF
```

**Step 3: Commit**

```bash
git add backend/Dockerfile.worker backend/.dockerignore
git commit -m "feat(docker): add Dockerfile for cloud worker with SAM3 model"
```

---

## Task 8: Update SAM3 Tracker for Local-First Model Loading

**Files:**
- Modify: `backend/app/ml/sam3_tracker.py`

**Step 1: Add local-first model loading**

In `sam3_tracker.py`, modify the `_load_predictor` method. Find:

```python
self._model = Sam3VideoModel.from_pretrained(
    "facebook/sam3",
    torch_dtype=dtype,
).to(self._device)

self._processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
```

Replace with:

```python
# Try local model path first (for Docker), then HuggingFace
model_path = self._find_model_path()
logger.info(f"Loading SAM3 from: {model_path}")

self._model = Sam3VideoModel.from_pretrained(
    model_path,
    torch_dtype=dtype,
    local_files_only=(model_path != "facebook/sam3"),
).to(self._device)

self._processor = Sam3VideoProcessor.from_pretrained(
    model_path,
    local_files_only=(model_path != "facebook/sam3"),
)
```

**Step 2: Add _find_model_path method**

Add this method to the `SAM3VideoTracker` class:

```python
def _find_model_path(self) -> str:
    """Find SAM3 model path, checking local paths first.

    Returns:
        Path to model (local path or HuggingFace model ID).
    """
    import os

    # Check paths in order of preference
    local_paths = [
        os.environ.get("CLOUD_MODEL_PATH", ""),  # Docker container
        "/models/sam3",  # Docker default
        str(Path.home() / ".cache/huggingface/hub/models--facebook--sam3/snapshots"),
    ]

    for path in local_paths:
        if path and Path(path).exists():
            # For HF cache, find the actual snapshot
            if "snapshots" in path:
                snapshots = list(Path(path).iterdir())
                if snapshots:
                    return str(snapshots[0])
            return path

    # Fall back to HuggingFace download
    return "facebook/sam3"
```

**Step 3: Commit**

```bash
git add backend/app/ml/sam3_tracker.py
git commit -m "feat(ml): add local-first model loading for Docker support"
```

---

## Task 9: End-to-End Test (Manual)

This task requires R2 to be configured. Skip if not ready.

**Step 1: Configure R2**

1. Create Cloudflare account and R2 bucket named `basketball-analyzer`
2. Create API token with R2 read/write permissions
3. Add to `.env`:
   ```
   R2_ACCOUNT_ID=your-account-id
   R2_ACCESS_KEY_ID=your-key
   R2_SECRET_ACCESS_KEY=your-secret
   R2_BUCKET_NAME=basketball-analyzer
   ```

**Step 2: Submit test job**

```bash
cd backend
python -m worker.cli submit --video-id 1 --video-path ../path/to/test-video.mp4
```

**Step 3: Check status**

```bash
python -m worker.cli status
```

**Step 4: Run cloud worker locally (for testing)**

```bash
python -m worker --cloud
```

**Step 5: Import results**

```bash
python -m worker.cli import-all
```

**Step 6: Verify in database**

```bash
sqlite3 basketball_analyzer.db "SELECT COUNT(*) FROM player_detections WHERE video_id=1"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | pyproject.toml | Add boto3, click dependencies |
| 2 | config.py | Add R2 configuration settings |
| 3 | cloud_storage.py | R2 upload/download/list operations |
| 4 | cli.py | CLI: submit, status, import commands |
| 5 | cloud_worker.py | Cloud worker that polls R2 |
| 6 | __main__.py | Add --cloud flag |
| 7 | Dockerfile.worker | Container with SAM3 model |
| 8 | sam3_tracker.py | Local-first model loading |
| 9 | (manual) | End-to-end test |

Total: ~8 implementation tasks + 1 manual test
