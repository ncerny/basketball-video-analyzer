# backend/worker/cli.py
"""CLI for cloud job management: submit, import, status."""

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
        raise click.ClickException("R2 not configured. Set R2_ACCOUNT_ID in .env")
    return CloudStorage(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
    )


def parse_job_id_from_key(key: str) -> str | None:
    """Safely parse job ID from S3 key like 'jobs/{job_id}.json'.

    Returns None if the key format is invalid.
    """
    parts = key.split("/")
    if len(parts) < 2:
        return None
    job_part = parts[1]
    if not job_part.endswith(".json"):
        return None
    return job_part.replace(".json", "")


@click.group()
def cli():
    """Cloud GPU job management for basketball video analysis."""
    pass


@cli.command()
@click.option("--video-id", required=True, type=int, help="Database video ID")
@click.option("--video-path", required=True, type=click.Path(exists=True), help="Path to video file")
@click.option("--sample-interval", default=1, type=int, help="Process every Nth frame")
@click.option("--confidence", default=0.25, type=float, help="Confidence threshold")
@click.option("--force", is_flag=True, help="Force new upload even if existing job found")
def submit(video_id: int, video_path: str, sample_interval: int, confidence: float, force: bool):
    """Submit a video for cloud GPU processing."""
    import asyncio
    from sqlalchemy import select
    from app.database import async_session_maker
    from app.models.video import Video

    # Validate video exists in database before upload
    async def check_video_exists():
        async with async_session_maker() as session:
            result = await session.execute(select(Video).where(Video.id == video_id))
            return result.scalar_one_or_none()

    try:
        video = asyncio.run(check_video_exists())
    except Exception as e:
        raise click.ClickException(f"Database error checking video: {e}")

    if not video:
        raise click.ClickException(f"Video {video_id} not found in database")

    storage = get_storage()
    video_path = Path(video_path)

    # Check for existing jobs that can be reused
    if not force:
        existing_jobs = storage.find_jobs_for_video(video_id)
        for existing in existing_jobs:
            # Only reuse failed or pending jobs with valid video
            if existing.status not in ("failed", "pending"):
                continue

            video_key = existing.parameters.get("video_key")
            if not video_key:
                continue

            if storage.video_exists(video_key):
                click.echo(f"Found existing job {existing.job_id} with video already uploaded")
                click.echo(f"Resetting job to pending (use --force to create new job)...")

                # Reset the job
                existing.status = "pending"
                existing.error = None
                existing.started_at = None
                existing.completed_at = None
                existing.frames_processed = 0
                existing.worker_id = None
                existing.last_heartbeat = None
                existing.parameters["sample_interval"] = sample_interval
                existing.parameters["confidence_threshold"] = confidence
                storage.upload_job_manifest(existing)

                click.echo(f"Job reset: {existing.job_id}")
                job_id = existing.job_id
                break
        else:
            # No reusable job found, create new one
            job_id = None
    else:
        job_id = None

    # Create new job if needed
    if job_id is None:
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

    # Auto-start RunPod if configured
    from worker.runpod_service import get_runpod_service
    runpod = get_runpod_service()
    if runpod._api_key:
        click.echo("Checking RunPod status...")
        if runpod.is_pod_running():
            click.echo("RunPod worker is already running - job will be processed shortly")
        else:
            click.echo("Starting RunPod worker...")
            if runpod.start_pod():
                click.echo("RunPod worker starting - job will be processed once pod is ready")
            else:
                click.echo("Failed to start RunPod - start manually or check RUNPOD_API_KEY")
    else:
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
        job_id = parse_job_id_from_key(key)
        if not job_id:
            continue
        manifest = storage.get_job_manifest(job_id)
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
    from app.database import async_session_maker
    from worker.job_importer import import_job_results

    storage = get_storage()

    # Check job status
    manifest = storage.get_job_manifest(job_id)
    if not manifest:
        raise click.ClickException(f"Job {job_id} not found")

    if manifest.status != "completed":
        raise click.ClickException(f"Job {job_id} is not completed (status: {manifest.status})")

    async def do_import():
        async with async_session_maker() as session:
            return await import_job_results(storage, manifest, session, cleanup=cleanup)

    try:
        count = asyncio.run(do_import())
        click.echo(f"Imported {count} detections into database")
        if cleanup:
            click.echo("Cleanup complete")
    except ValueError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Import error: {e}")


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
        job_id = parse_job_id_from_key(key)
        if not job_id:
            continue
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


@cli.command("gpu-status")
def gpu_status():
    """Show RunPod GPU worker status."""
    from worker.runpod_service import get_runpod_service

    runpod = get_runpod_service()
    status = runpod.get_status_summary()

    if not status.get("enabled"):
        click.echo(f"RunPod: disabled ({status.get('reason', 'unknown')})")
        click.echo("Set RUNPOD_API_KEY environment variable to enable auto-scaling")
        return

    if not status.get("pod_found"):
        click.echo(f"RunPod: enabled but no pod found with name '{status.get('template_name')}'")
        click.echo("Create a pod in RunPod console with this name first")
        return

    click.echo(f"RunPod Pod: {status['pod_name']}")
    click.echo(f"  Status: {status['status']}")
    click.echo(f"  GPU: {status.get('gpu_type', 'unknown')}")
    click.echo(f"  Cost: ${status.get('cost_per_hour', 0):.2f}/hour")
    click.echo(f"  Pod ID: {status['pod_id']}")


@cli.command("gpu-start")
def gpu_start():
    """Start the RunPod GPU worker."""
    from worker.runpod_service import get_runpod_service

    runpod = get_runpod_service()
    if not runpod._api_key:
        raise click.ClickException("RUNPOD_API_KEY not set")

    if runpod.is_pod_running():
        click.echo("RunPod worker is already running")
        return

    click.echo("Starting RunPod worker...")
    if runpod.start_pod():
        click.echo("Start command sent. Pod will be ready in ~30-60 seconds.")
    else:
        raise click.ClickException("Failed to start pod. Check logs for details.")


@cli.command("gpu-stop")
def gpu_stop():
    """Stop the RunPod GPU worker."""
    from worker.runpod_service import get_runpod_service

    runpod = get_runpod_service()
    if not runpod._api_key:
        raise click.ClickException("RUNPOD_API_KEY not set")

    click.echo("Stopping RunPod worker...")
    if runpod.stop_pod():
        click.echo("Stop command sent. Pod will stop shortly.")
    else:
        raise click.ClickException("Failed to stop pod. Check logs for details.")


if __name__ == "__main__":
    cli()
