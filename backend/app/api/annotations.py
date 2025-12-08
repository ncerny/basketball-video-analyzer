"""Annotations API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.annotation import Annotation as AnnotationModel, AnnotationType, CreatedBy
from app.models.annotation_video import AnnotationVideo as AnnotationVideoModel
from app.models.game import Game as GameModel
from app.models.video import Video as VideoModel
from app.schemas.annotation import (
    Annotation,
    AnnotationCreate,
    AnnotationList,
    AnnotationUpdate,
    AnnotationVideoLink,
)

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("", response_model=Annotation, status_code=status.HTTP_201_CREATED)
async def create_annotation(
    annotation_data: AnnotationCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Annotation:
    """Create a new annotation with video links."""
    # Verify game exists
    game_stmt = select(GameModel).where(GameModel.id == annotation_data.game_id)
    game_result = await db.execute(game_stmt)
    game = game_result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {annotation_data.game_id} not found",
        )

    # Validate annotation_type
    try:
        annotation_type_enum = AnnotationType(annotation_data.annotation_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid annotation_type: {annotation_data.annotation_type}. Must be one of: play, event, note",
        )

    # Validate created_by
    try:
        created_by_enum = CreatedBy(annotation_data.created_by.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid created_by: {annotation_data.created_by}. Must be one of: ai, user",
        )

    # Validate timestamps
    if annotation_data.game_timestamp_end <= annotation_data.game_timestamp_start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="game_timestamp_end must be greater than game_timestamp_start",
        )

    # Verify videos exist if provided
    if annotation_data.video_ids:
        video_stmt = select(VideoModel).where(VideoModel.id.in_(annotation_data.video_ids))
        video_result = await db.execute(video_stmt)
        videos = video_result.scalars().all()

        if len(videos) != len(annotation_data.video_ids):
            found_ids = {v.id for v in videos}
            missing_ids = set(annotation_data.video_ids) - found_ids
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Videos not found: {missing_ids}",
            )

    # Create annotation
    annotation = AnnotationModel(
        game_id=annotation_data.game_id,
        game_timestamp_start=annotation_data.game_timestamp_start,
        game_timestamp_end=annotation_data.game_timestamp_end,
        annotation_type=annotation_type_enum,
        confidence_score=annotation_data.confidence_score,
        verified=annotation_data.verified,
        created_by=created_by_enum,
    )
    db.add(annotation)
    await db.flush()  # Get annotation ID

    # Create video links (for now, use simple 1:1 mapping of game time to video time)
    # TODO: In a real implementation, calculate actual video timestamps based on game_time_offset
    for video_id in annotation_data.video_ids:
        video_link = AnnotationVideoModel(
            annotation_id=annotation.id,
            video_id=video_id,
            video_timestamp_start=annotation_data.game_timestamp_start,
            video_timestamp_end=annotation_data.game_timestamp_end,
        )
        db.add(video_link)

    await db.commit()
    await db.refresh(annotation)

    # Load video links
    stmt = (
        select(AnnotationModel)
        .options(selectinload(AnnotationModel.annotation_videos))
        .where(AnnotationModel.id == annotation.id)
    )
    result = await db.execute(stmt)
    annotation_with_links = result.scalar_one()

    return _build_annotation_response(annotation_with_links)


@router.get("", response_model=AnnotationList)
async def list_annotations(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    game_id: Annotated[int | None, Query(description="Filter by game ID")] = None,
    annotation_type: Annotated[str | None, Query(description="Filter by type")] = None,
    created_by: Annotated[str | None, Query(description="Filter by creator (ai/user)")] = None,
    verified: Annotated[bool | None, Query(description="Filter by verified status")] = None,
) -> AnnotationList:
    """List all annotations with pagination and optional filtering."""
    # Build query
    stmt = select(AnnotationModel).options(selectinload(AnnotationModel.annotation_videos))

    # Apply filters
    if game_id is not None:
        stmt = stmt.where(AnnotationModel.game_id == game_id)

    if annotation_type is not None:
        try:
            type_enum = AnnotationType(annotation_type.lower())
            stmt = stmt.where(AnnotationModel.annotation_type == type_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid annotation_type: {annotation_type}. Must be one of: play, event, note",
            )

    if created_by is not None:
        try:
            creator_enum = CreatedBy(created_by.lower())
            stmt = stmt.where(AnnotationModel.created_by == creator_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid created_by: {created_by}. Must be one of: ai, user",
            )

    if verified is not None:
        stmt = stmt.where(AnnotationModel.verified == verified)

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination and ordering
    stmt = (
        stmt.order_by(AnnotationModel.game_id, AnnotationModel.game_timestamp_start)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    # Execute query
    result = await db.execute(stmt)
    annotations = result.scalars().all()

    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size

    return AnnotationList(
        annotations=[_build_annotation_response(a) for a in annotations],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{annotation_id}", response_model=Annotation)
async def get_annotation(
    annotation_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Annotation:
    """Get an annotation by ID with video links."""
    stmt = (
        select(AnnotationModel)
        .options(selectinload(AnnotationModel.annotation_videos))
        .where(AnnotationModel.id == annotation_id)
    )
    result = await db.execute(stmt)
    annotation = result.scalar_one_or_none()

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation with id {annotation_id} not found",
        )

    return _build_annotation_response(annotation)


@router.patch("/{annotation_id}", response_model=Annotation)
async def update_annotation(
    annotation_id: int,
    annotation_data: AnnotationUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Annotation:
    """Update an annotation."""
    stmt = (
        select(AnnotationModel)
        .options(selectinload(AnnotationModel.annotation_videos))
        .where(AnnotationModel.id == annotation_id)
    )
    result = await db.execute(stmt)
    annotation = result.scalar_one_or_none()

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation with id {annotation_id} not found",
        )

    # Update only provided fields
    update_data = annotation_data.model_dump(exclude_unset=True, exclude={"video_ids"})

    # Validate and convert enums
    if "annotation_type" in update_data and update_data["annotation_type"] is not None:
        try:
            update_data["annotation_type"] = AnnotationType(update_data["annotation_type"].lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid annotation_type. Must be one of: play, event, note",
            )

    if "created_by" in update_data and update_data["created_by"] is not None:
        try:
            update_data["created_by"] = CreatedBy(update_data["created_by"].lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid created_by. Must be one of: ai, user",
            )

    # Validate timestamps if both provided
    start = update_data.get("game_timestamp_start", annotation.game_timestamp_start)
    end = update_data.get("game_timestamp_end", annotation.game_timestamp_end)
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="game_timestamp_end must be greater than game_timestamp_start",
        )

    for field, value in update_data.items():
        setattr(annotation, field, value)

    # Update video links if provided
    if annotation_data.video_ids is not None:
        # Delete existing links
        delete_stmt = AnnotationVideoModel.__table__.delete().where(
            AnnotationVideoModel.annotation_id == annotation_id
        )
        await db.execute(delete_stmt)

        # Create new links
        for video_id in annotation_data.video_ids:
            video_link = AnnotationVideoModel(
                annotation_id=annotation.id,
                video_id=video_id,
                video_timestamp_start=annotation.game_timestamp_start,
                video_timestamp_end=annotation.game_timestamp_end,
            )
            db.add(video_link)

    await db.commit()
    await db.refresh(annotation)

    # Reload with video links
    stmt = (
        select(AnnotationModel)
        .options(selectinload(AnnotationModel.annotation_videos))
        .where(AnnotationModel.id == annotation_id)
    )
    result = await db.execute(stmt)
    annotation_with_links = result.scalar_one()

    return _build_annotation_response(annotation_with_links)


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_annotation(
    annotation_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete an annotation and all associated video links."""
    stmt = select(AnnotationModel).where(AnnotationModel.id == annotation_id)
    result = await db.execute(stmt)
    annotation = result.scalar_one_or_none()

    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation with id {annotation_id} not found",
        )

    await db.delete(annotation)
    await db.commit()


def _build_annotation_response(annotation: AnnotationModel) -> Annotation:
    """Build annotation response with video links."""
    video_links = [
        AnnotationVideoLink(
            video_id=link.video_id,
            video_timestamp_start=link.video_timestamp_start,
            video_timestamp_end=link.video_timestamp_end,
        )
        for link in annotation.annotation_videos
    ]

    return Annotation(
        id=annotation.id,
        game_id=annotation.game_id,
        game_timestamp_start=annotation.game_timestamp_start,
        game_timestamp_end=annotation.game_timestamp_end,
        annotation_type=annotation.annotation_type.value,
        confidence_score=annotation.confidence_score,
        verified=annotation.verified,
        created_by=annotation.created_by.value,
        created_at=annotation.created_at,
        updated_at=annotation.updated_at,
        video_links=video_links,
    )
