"""Video upload API endpoints."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.game import Game as GameModel
from app.models.video import ProcessingStatus, Video as VideoModel
from app.schemas.video import Video
from app.services.video_storage import VideoStorageService

router = APIRouter(prefix="/video-upload", tags=["video-upload"])


@router.post("", response_model=Video, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: Annotated[UploadFile, File(description="Video file to upload")],
    game_id: Annotated[int, Form(description="ID of the game this video belongs to")],
    db: Annotated[AsyncSession, Depends(get_db)],
    sequence_order: Annotated[int | None, Form(description="Optional sequence order")] = None,
) -> Video:
    """Upload a video file and create a video record.

    This endpoint:
    1. Validates the game exists
    2. Validates the file is a video
    3. Saves the video to filesystem storage
    4. Extracts metadata using FFmpeg (duration, fps, resolution, recorded_at)
    5. Creates a video database record

    The video processing_status will be set to 'completed' if metadata extraction succeeds.
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

    # Initialize storage service
    storage_service = VideoStorageService()

    try:
        # Save video file
        relative_path = await storage_service.save_video(file, game_id)

        # Extract metadata
        metadata = storage_service.extract_metadata(relative_path)

        # Create video record
        video = VideoModel(
            game_id=game_id,
            file_path=relative_path,
            duration_seconds=metadata["duration_seconds"],
            fps=metadata["fps"],
            resolution=metadata["resolution"],
            recorded_at=metadata.get("recorded_at"),
            sequence_order=sequence_order,
            processing_status=ProcessingStatus.COMPLETED,
            processed=True,
        )

        db.add(video)
        await db.commit()
        await db.refresh(video)

        return Video.model_validate(video)

    except ValueError as e:
        # Metadata extraction failed - delete uploaded file and return error
        try:
            storage_service.delete_video(relative_path)
        except FileNotFoundError:
            pass

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract video metadata: {str(e)}",
        )

    except Exception as e:
        # Unexpected error - try to cleanup
        try:
            if "relative_path" in locals():
                storage_service.delete_video(relative_path)
        except FileNotFoundError:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video upload failed: {str(e)}",
        )
