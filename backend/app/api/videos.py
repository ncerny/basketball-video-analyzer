"""Videos API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.video import ProcessingStatus, Video as VideoModel
from app.schemas.video import Video, VideoCreate, VideoList, VideoUpdate
from app.services.thumbnail_generator import ThumbnailGeneratorService

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=Video, status_code=status.HTTP_201_CREATED)
async def create_video(
    video_data: VideoCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Video:
    """Create a new video record."""
    # Verify game exists
    from app.models.game import Game as GameModel

    game_stmt = select(GameModel).where(GameModel.id == video_data.game_id)
    game_result = await db.execute(game_stmt)
    game = game_result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {video_data.game_id} not found",
        )

    video = VideoModel(**video_data.model_dump())
    db.add(video)
    await db.commit()
    await db.refresh(video)

    return Video.model_validate(video)


@router.get("", response_model=VideoList)
async def list_videos(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    game_id: Annotated[int | None, Query(description="Filter by game ID")] = None,
    processing_status: Annotated[
        str | None, Query(description="Filter by processing status")
    ] = None,
    processed: Annotated[bool | None, Query(description="Filter by processed flag")] = None,
) -> VideoList:
    """List all videos with pagination and optional filtering."""
    # Build query
    stmt = select(VideoModel)

    # Apply filters
    if game_id is not None:
        stmt = stmt.where(VideoModel.game_id == game_id)
    if processing_status is not None:
        try:
            status_enum = ProcessingStatus(processing_status.lower())
            stmt = stmt.where(VideoModel.processing_status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing_status: {processing_status}. Must be one of: pending, processing, completed, failed",
            )
    if processed is not None:
        stmt = stmt.where(VideoModel.processed == processed)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination and ordering
    stmt = (
        stmt.order_by(VideoModel.game_id, VideoModel.sequence_order.nullsfirst())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    # Execute query
    result = await db.execute(stmt)
    videos = result.scalars().all()

    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size

    return VideoList(
        videos=[Video.model_validate(v) for v in videos],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{video_id}", response_model=Video)
async def get_video(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Video:
    """Get a video by ID."""
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    return Video.model_validate(video)


@router.patch("/{video_id}", response_model=Video)
async def update_video(
    video_id: int,
    video_data: VideoUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Video:
    """Update a video (typically used to update processing status and timeline data)."""
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    # Update only provided fields
    update_data = video_data.model_dump(exclude_unset=True)

    # Convert processing_status string to enum if provided
    if "processing_status" in update_data and update_data["processing_status"] is not None:
        try:
            update_data["processing_status"] = ProcessingStatus(
                update_data["processing_status"].lower()
            )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing_status: {update_data['processing_status']}. Must be one of: pending, processing, completed, failed",
            )

    for field, value in update_data.items():
        setattr(video, field, value)

    await db.commit()
    await db.refresh(video)

    return Video.model_validate(video)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a video and all associated data (detections, annotation links)."""
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    await db.delete(video)
    await db.commit()


@router.post("/{video_id}/thumbnail", response_model=Video)
async def generate_thumbnail(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    timestamp: Annotated[
        float | None,
        Query(ge=0, description="Timestamp in seconds to extract frame from (default: middle of video)"),
    ] = None,
) -> Video:
    """Generate a thumbnail for a video.

    Extracts a frame from the video at the specified timestamp (or middle if not specified)
    and saves it as a JPEG thumbnail. Updates the video record with the thumbnail path.
    """
    # Get video
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    # Generate thumbnail
    thumbnail_service = ThumbnailGeneratorService()
    try:
        thumbnail_path = thumbnail_service.generate_thumbnail(
            video.file_path, video.game_id, timestamp
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate thumbnail: {str(e)}",
        )

    # Update video record with thumbnail path
    video.thumbnail_path = thumbnail_path
    await db.commit()
    await db.refresh(video)

    return Video.model_validate(video)


@router.get("/{video_id}/thumbnail")
async def get_thumbnail(
    video_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    """Get the thumbnail image for a video.

    Returns the thumbnail image file if it exists. If no thumbnail exists,
    returns a 404 error. Use POST /videos/{video_id}/thumbnail to generate one first.
    """
    # Get video
    stmt = select(VideoModel).where(VideoModel.id == video_id)
    result = await db.execute(stmt)
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with id {video_id} not found",
        )

    if not video.thumbnail_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No thumbnail exists for video {video_id}. Use POST /videos/{video_id}/thumbnail to generate one.",
        )

    # Get absolute path to thumbnail
    thumbnail_service = ThumbnailGeneratorService()
    thumbnail_absolute_path = thumbnail_service.get_absolute_path(video.thumbnail_path)

    if not thumbnail_service.thumbnail_exists(video.thumbnail_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thumbnail file not found at {video.thumbnail_path}",
        )

    return FileResponse(
        path=str(thumbnail_absolute_path),
        media_type="image/jpeg",
        filename=f"video_{video_id}_thumbnail.jpg",
    )
