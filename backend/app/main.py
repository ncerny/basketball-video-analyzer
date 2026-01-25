"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
from app.api import (
    annotations,
    detection,
    game_rosters,
    games,
    players,
    timeline,
    video_upload,
    videos,
)

app = FastAPI(
    title="Basketball Video Analyzer API",
    description="API for analyzing youth basketball game videos with player tracking and play tagging",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(games.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(game_rosters.router, prefix="/api")
app.include_router(video_upload.router, prefix="/api")
app.include_router(timeline.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(detection.router, prefix="/api")


async def monitor_cloud_jobs() -> None:
    """Background task to manage cloud GPU jobs.

    - Imports completed job results into the database
    - Stops RunPod pods when no jobs remain
    """
    import asyncio

    logger = logging.getLogger(__name__)

    if not settings.r2_account_id:
        logger.info("Cloud job monitoring disabled (no R2 config)")
        return

    from app.database import async_session_maker
    from worker.cloud_storage import CloudStorage
    from worker.job_importer import import_job_results

    # Only import runpod if configured
    runpod = None
    if settings.runpod_api_key:
        from worker.runpod_service import get_runpod_service
        runpod = get_runpod_service()

    while True:
        await asyncio.sleep(settings.runpod_idle_check_interval)

        try:
            storage = CloudStorage(
                account_id=settings.r2_account_id,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_bucket_name,
            )

            # Scan all jobs
            has_work = False
            completed_jobs = []
            job_statuses = []

            response = storage._client.list_objects_v2(
                Bucket=storage._bucket, Prefix="jobs/"
            )
            for obj in response.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                job_id = obj["Key"].replace("jobs/", "").replace(".json", "")
                manifest = storage.get_job_manifest(job_id)
                if not manifest:
                    continue

                job_statuses.append(f"{job_id[:8]}:{manifest.status}")

                if manifest.status in ("pending", "processing"):
                    has_work = True
                elif manifest.status == "completed":
                    completed_jobs.append(manifest)

            if job_statuses:
                logger.info(f"Cloud jobs: {', '.join(job_statuses)}")

            # Import completed jobs
            for manifest in completed_jobs:
                try:
                    async with async_session_maker() as session:
                        count = await import_job_results(
                            storage, manifest, session, cleanup=True
                        )
                        logger.info(
                            f"Auto-imported job {manifest.job_id}: "
                            f"{count} detections for video {manifest.video_id}"
                        )
                except Exception as e:
                    logger.error(f"Failed to auto-import job {manifest.job_id}: {e}")

            # Stop pod if no work remains
            if not has_work:
                if not runpod:
                    logger.info("No work remaining, but RUNPOD_API_KEY not configured")
                else:
                    pod_running = runpod.is_pod_running()
                    logger.info(f"No work remaining, pod running: {pod_running}")
                    if pod_running:
                        logger.info("Stopping RunPod to save costs...")
                        if runpod.stop_pod():
                            logger.info("RunPod pod stop requested")
                        else:
                            logger.warning("Failed to stop RunPod pod")

        except Exception as e:
            logger.warning(f"Error in cloud job monitor: {e}")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on application startup."""
    from app.services.detection_pipeline import create_detection_job_worker
    from app.services.job_manager import get_job_manager
    import asyncio

    # Register detection worker in background (don't block startup)
    job_manager = get_job_manager()
    asyncio.create_task(create_detection_job_worker(job_manager))

    # Start cloud job monitoring (auto-import results, stop idle pods)
    asyncio.create_task(monitor_cloud_jobs())


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning API information."""
    return {
        "name": "Basketball Video Analyzer API",
        "version": __version__,
        "status": "running",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
