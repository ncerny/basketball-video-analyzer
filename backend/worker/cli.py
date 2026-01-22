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
