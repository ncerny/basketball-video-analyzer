"""Video upload API endpoints."""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.game import Game as GameModel
from app.models.video import ProcessingStatus, Video as VideoModel
from app.schemas.video import Video
from app.services.cloud_job_service import get_cloud_job_service
from app.services.r2_storage import get_r2_storage_service
from app.services.video_storage import VideoStorageService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-upload", tags=["video-upload"])


@router.post("", response_model=Video, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: Annotated[UploadFile, File(description="Video file to upload")],
    game_id: Annotated[int, Form(description="ID of the game this video belongs to")],
    db: Annotated[AsyncSession, Depends(get_db)],
    sequence_order: Annotated[int | None, Form(description="Optional sequence order")] = None,
    auto_process: Annotated[bool, Form(description="Auto-submit cloud processing job")] = True,
) -> Video:
    """Upload a video file and create a video record.

    This endpoint:
    1. Validates the game exists
    2. Validates the file is a video
    3. Saves the video to temp local storage
    4. Extracts metadata using FFmpeg (duration, fps, resolution, recorded_at)
    5. Uploads video to R2 cloud storage
    6. Deletes local temp file
    7. Creates a video database record with r2_key
    8. Optionally submits cloud processing job

    The video processing_status will be set to 'pending' for cloud processing.
    """
    # Verify game exists
    game_stmt = select(GameModel).where(GameModel.id == game_id)
    game_result = await db.execute(game_stmt)
    game = game_result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {game_id} not found",
        )

    # Validate file type (check both content type and extension)
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    file_extension = Path(file.filename or "").suffix.lower()

    is_valid = False
    if file.content_type and file.content_type.startswith("video/"):
        is_valid = True
    elif file_extension in valid_extensions:
        is_valid = True

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected video file but got {file.content_type} with extension {file_extension}",
        )

    # Initialize services
    storage_service = VideoStorageService()
    r2_service = get_r2_storage_service()
    cloud_job_service = get_cloud_job_service()

    relative_path = None
    r2_key = None

    try:
        # Save video file to local temp storage (for FFmpeg metadata extraction)
        relative_path = await storage_service.save_video(file, game_id)

        # Extract metadata using FFmpeg
        metadata = storage_service.extract_metadata(relative_path)

        # Upload to R2 if configured
        if r2_service.is_configured:
            local_path = storage_service.get_absolute_path(relative_path)
            filename = Path(relative_path).name
            r2_key = r2_service.upload_video(local_path, game_id, filename)

            # Delete local file after successful R2 upload
            try:
                storage_service.delete_video(relative_path)
                logger.info(f"Deleted local temp file: {relative_path}")
            except Exception as e:
                logger.warning(f"Failed to delete local temp file: {e}")

        # Create video record
        video = VideoModel(
            game_id=game_id,
            file_path=relative_path,  # Keep for legacy/fallback
            r2_key=r2_key,
            duration_seconds=metadata["duration_seconds"],
            fps=metadata["fps"],
            resolution=metadata["resolution"],
            recorded_at=metadata.get("recorded_at"),
            sequence_order=sequence_order,
            processing_status=ProcessingStatus.PENDING if r2_key else ProcessingStatus.COMPLETED,
            processed=not bool(r2_key),  # Not processed if going to cloud
        )

        db.add(video)
        await db.commit()
        await db.refresh(video)

        # Submit cloud processing job if R2 configured and auto_process enabled
        if r2_key and auto_process and cloud_job_service.is_configured:
            try:
                job_id = cloud_job_service.submit_job(
                    video_id=video.id,
                    r2_key=r2_key,
                )
                logger.info(f"Submitted cloud job {job_id} for video {video.id}")
            except Exception as e:
                # Don't fail upload if job submission fails
                logger.warning(f"Failed to submit cloud job for video {video.id}: {e}")

        return Video.model_validate(video)

    except ValueError as e:
        # Metadata extraction failed - cleanup and return error
        if relative_path:
            try:
                storage_service.delete_video(relative_path)
            except FileNotFoundError:
                pass
        if r2_key:
            try:
                r2_service.delete_video(r2_key)
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract video metadata: {str(e)}",
        )

    except Exception as e:
        # Unexpected error - cleanup
        if relative_path:
            try:
                storage_service.delete_video(relative_path)
            except FileNotFoundError:
                pass
        if r2_key:
            try:
                r2_service.delete_video(r2_key)
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video upload failed: {str(e)}",
        )
