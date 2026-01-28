"""Detection API endpoints for player detection and job management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.detection import PlayerDetection
from app.models.video import Video as VideoModel
from app.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionJobRequest,
    DetectionJobResponse,
    DetectionStats,
    JobResponse,
    JobProgress,
    TrackReprocessRequest,
    TrackReprocessResponse,
    VideoDetectionsResponse,
)

router = APIRouter(tags=["detection"])


@router.post(
    "/videos/{video_id}/detect",
    response_model=DetectionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_detection(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: DetectionJobRequest | None = None,
) -> DetectionJobResponse:
    """Start player detection for a video.

    Submits a background job to detect players in the video frames.
    Returns immediately with a job ID that can be used to poll status.

    Jobs are queued in the database for processing by an external worker process.
    """
    from app.services.job_service import create_detection_job

    # Verify video exists
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    # Get request parameters or defaults
    params = request if request is not None else DetectionJobRequest()

    job = await create_detection_job(
        db,
        video_id=video_id,
        parameters={
            "sample_interval": params.sample_interval,
            "batch_size": params.batch_size,
            "confidence_threshold": params.confidence_threshold,
            "max_seconds": params.max_seconds,
            "enable_court_detection": params.enable_court_detection,
            "enable_jersey_ocr": settings.enable_jersey_ocr,
        },
    )
    await db.commit()

    return DetectionJobResponse(
        job_id=job.id,
        video_id=video_id,
        message=f"Detection job queued for worker. Poll GET /api/jobs/{job.id} for status.",
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobResponse:
    """Get the status of a background job.

    Use this to poll for job completion after starting detection.
    When status is 'completed', the result field contains detection statistics.
    """
    from app.services.job_service import get_job

    db_job = await get_job(db, job_id)
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found",
        )

    return JobResponse(
        id=db_job.id,
        job_type=db_job.job_type.value,
        status=db_job.status.value,
        progress=JobProgress(
            current=db_job.progress_current,
            total=db_job.progress_total,
            percentage=db_job.progress_percentage,
            message=db_job.progress_message,
        ),
        result=db_job.result,
        error=db_job.error_message,
        created_at=db_job.created_at,
        started_at=db_job.started_at,
        completed_at=db_job.completed_at,
        metadata=db_job.parameters,
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Cancel a pending job.

    Only jobs in 'pending' status can be cancelled. Jobs that are already
    processing cannot be cancelled via API (would need to stop the worker).
    """
    from app.models.processing_job import JobStatus as DBJobStatus
    from app.services.job_service import cancel_job as cancel_db_job, get_job

    db_job = await get_job(db, job_id)
    if not db_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found",
        )

    if db_job.status in (DBJobStatus.COMPLETED, DBJobStatus.FAILED, DBJobStatus.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {db_job.status.value}",
        )
    if db_job.status == DBJobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel job that is already processing. Stop the worker to abort.",
        )

    cancelled = await cancel_db_job(db, job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job",
        )
    await db.commit()


@router.get("/videos/{video_id}/detections", response_model=VideoDetectionsResponse)
async def get_video_detections(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    frame_start: Annotated[
        int | None, Query(ge=0, description="Filter detections from this frame")
    ] = None,
    frame_end: Annotated[
        int | None, Query(ge=0, description="Filter detections up to this frame")
    ] = None,
    min_confidence: Annotated[
        float | None, Query(ge=0, le=1, description="Minimum confidence threshold")
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=500000, description="Maximum detections to return")
    ] = 50000,
) -> VideoDetectionsResponse:
    """Get all player detections for a video.

    Returns detection results from completed detection jobs.
    Use filters to narrow results by frame range or confidence.
    """
    # Verify video exists
    video_stmt = select(VideoModel).where(VideoModel.id == video_id)
    video_result = await db.execute(video_stmt)
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    # Build query for detections
    stmt = select(PlayerDetection).where(PlayerDetection.video_id == video_id)

    if frame_start is not None:
        stmt = stmt.where(PlayerDetection.frame_number >= frame_start)
    if frame_end is not None:
        stmt = stmt.where(PlayerDetection.frame_number <= frame_end)
    if min_confidence is not None:
        stmt = stmt.where(PlayerDetection.confidence_score >= min_confidence)

    stmt = stmt.order_by(PlayerDetection.frame_number, PlayerDetection.tracking_id).limit(limit)

    result = await db.execute(stmt)
    detections = result.scalars().all()

    # Get count of unique frames
    frames_stmt = select(func.count(func.distinct(PlayerDetection.frame_number))).where(
        PlayerDetection.video_id == video_id
    )
    frames_result = await db.execute(frames_stmt)
    frames_count = frames_result.scalar_one()

    # Get total count (without limit)
    count_stmt = select(func.count()).select_from(
        select(PlayerDetection).where(PlayerDetection.video_id == video_id).subquery()
    )
    count_result = await db.execute(count_stmt)
    total_count = count_result.scalar_one()

    return VideoDetectionsResponse(
        video_id=video_id,
        total_detections=total_count,
        detections=[
            Detection(
                id=d.id,
                video_id=d.video_id,
                frame_number=d.frame_number,
                player_id=d.player_id,
                bbox=BoundingBox(
                    x=d.bbox_x,
                    y=d.bbox_y,
                    width=d.bbox_width,
                    height=d.bbox_height,
                ),
                tracking_id=d.tracking_id,
                confidence_score=d.confidence_score,
            )
            for d in detections
        ],
        frames_with_detections=frames_count,
    )


@router.get("/videos/{video_id}/detections/stats", response_model=DetectionStats)
async def get_detection_stats(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DetectionStats:
    """Get detection statistics for a video.

    Returns aggregate statistics about detections without returning all detection data.
    """
    # Verify video exists
    video_stmt = select(VideoModel).where(VideoModel.id == video_id)
    video_result = await db.execute(video_stmt)
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    # Get total detections
    total_stmt = (
        select(func.count())
        .select_from(PlayerDetection)
        .where(PlayerDetection.video_id == video_id)
    )
    total_result = await db.execute(total_stmt)
    total_detections = total_result.scalar_one()

    # Get unique frames count
    frames_stmt = select(func.count(func.distinct(PlayerDetection.frame_number))).where(
        PlayerDetection.video_id == video_id
    )
    frames_result = await db.execute(frames_stmt)
    frames_with_detections = frames_result.scalar_one()

    # Calculate average detections per frame
    avg_per_frame = total_detections / frames_with_detections if frames_with_detections > 0 else 0.0

    # Note: We don't have class_id stored in PlayerDetection, so we estimate
    # For now, assume all detections are persons (balls are relatively rare)
    # A proper implementation would store class_id in PlayerDetection
    persons_detected = total_detections
    balls_detected = 0

    return DetectionStats(
        video_id=video_id,
        total_frames_processed=frames_with_detections,  # Approximation
        total_detections=total_detections,
        persons_detected=persons_detected,
        balls_detected=balls_detected,
        frames_with_detections=frames_with_detections,
        avg_detections_per_frame=round(avg_per_frame, 2),
    )


@router.delete("/videos/{video_id}/detections", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_detections(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete all detections for a video.

    Clears all detection data including player detections, jersey OCR results,
    and processing batch records. Use this before re-running detection.
    """
    from sqlalchemy import delete

    from app.models.jersey_number import JerseyNumber
    from app.models.processing_batch import ProcessingBatch

    video_stmt = select(VideoModel).where(VideoModel.id == video_id)
    video_result = await db.execute(video_stmt)
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    await db.execute(delete(JerseyNumber).where(JerseyNumber.video_id == video_id))
    await db.execute(delete(PlayerDetection).where(PlayerDetection.video_id == video_id))
    await db.execute(delete(ProcessingBatch).where(ProcessingBatch.video_id == video_id))
    await db.commit()


@router.post("/videos/{video_id}/reprocess-tracks", response_model=TrackReprocessResponse)
async def reprocess_tracks(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: TrackReprocessRequest | None = None,
) -> TrackReprocessResponse:
    """Reprocess tracks for an already-analyzed video.

    Runs identity switch detection and track merging on existing detection data.
    Use this to apply tracking improvements to videos processed before these
    features were added.

    - Identity switch detection: Splits tracks where jersey number changes mid-track
    - Track merging: Consolidates fragmented tracks based on spatial proximity and jersey numbers
    """
    from app.config import settings
    from app.services.identity_switch_detector import IdentitySwitchConfig, IdentitySwitchDetector
    from app.services.track_merger import TrackMerger, TrackMergerConfig

    video_stmt = select(VideoModel).where(VideoModel.id == video_id)
    video_result = await db.execute(video_stmt)
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    params = request if request is not None else TrackReprocessRequest()

    switches_detected = 0
    tracks_split = 0

    if params.enable_identity_switch_detection:
        switch_config = IdentitySwitchConfig(
            window_size_frames=settings.identity_switch_window_size_frames,
            min_readings_per_window=settings.identity_switch_min_readings,
            switch_threshold=settings.identity_switch_threshold,
        )
        detector = IdentitySwitchDetector(db, switch_config)
        switch_result = await detector.detect_and_split(video_id)
        switches_detected = switch_result.switches_detected
        tracks_split = switch_result.tracks_split

    tracks_before = 0
    tracks_after = 0
    spatial_merges = 0
    jersey_merges = 0

    if params.enable_track_merging:
        merger_config = TrackMergerConfig(
            enable_jersey_merge=settings.enable_jersey_merge,
            min_jersey_confidence=settings.min_jersey_confidence,
            min_jersey_readings=settings.min_jersey_readings,
        )
        merger = TrackMerger(db, merger_config)
        merge_result = await merger.merge_tracks(video_id)
        tracks_before = merge_result.original_track_count
        tracks_after = merge_result.merged_track_count
        spatial_merges = merge_result.spatial_merges
        jersey_merges = merge_result.jersey_merges

    return TrackReprocessResponse(
        video_id=video_id,
        identity_switches_detected=switches_detected,
        tracks_before_merge=tracks_before,
        tracks_after_merge=tracks_after,
        spatial_merges=spatial_merges,
        jersey_merges=jersey_merges,
        message=f"Reprocessed: {tracks_split} tracks split, {tracks_before} → {tracks_after} tracks after merge",
    )


@router.get("/ml-config")
async def get_ml_config() -> dict:
    """Get ML configuration and device information for diagnostics."""
    import torch

    from app.config import settings

    # Resolve device using same logic as SAM3 tracker
    if settings.ml_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = settings.ml_device

    torch_info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }

    return {
        "ml_device_setting": settings.ml_device,
        "resolved_device": resolved_device,
        "inference_timing_enabled": settings.enable_inference_timing,
        "sam3_prompt": settings.sam3_prompt,
        "sam3_confidence_threshold": settings.sam3_confidence_threshold,
        "sam3_use_half_precision": settings.sam3_use_half_precision,
        "sam3_memory_window_size": settings.sam3_memory_window_size,
        "sam3_use_torch_compile": settings.sam3_use_torch_compile,
        "torch": torch_info,
    }
