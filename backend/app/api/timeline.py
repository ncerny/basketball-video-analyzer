"""Timeline sequencing API endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.game import Game as GameModel
from app.services.timeline_sequencer import TimelineSequencer

router = APIRouter(prefix="/games/{game_id}/timeline", tags=["timeline"])


class TimelineGapResponse(BaseModel):
    """Response model for timeline gaps."""

    after_video_id: int
    before_video_id: int
    gap_seconds: float


class TimelineOverlapResponse(BaseModel):
    """Response model for timeline overlaps."""

    video1_id: int
    video2_id: int
    overlap_seconds: float


class SequencingResponse(BaseModel):
    """Response model for sequencing operation."""

    sequenced_count: int
    total_duration: float
    gaps: list[TimelineGapResponse]
    overlaps: list[TimelineOverlapResponse]
    warnings: list[str]


class TimelineSummaryResponse(BaseModel):
    """Response model for timeline summary."""

    game_id: int
    video_count: int
    total_duration: float
    sequenced_count: int
    unsequenced_count: int
    has_gaps: bool
    has_overlaps: bool


@router.post("/sequence", response_model=SequencingResponse, status_code=status.HTTP_200_OK)
async def sequence_game_timeline(
    game_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SequencingResponse:
    """Sequence all videos for a game in chronological order.

    This endpoint:
    - Sorts videos by recorded_at timestamp
    - Assigns sequence_order (0, 1, 2, ...)
    - Calculates game_time_offset for each video
    - Detects gaps and overlaps between videos

    Only videos with recorded_at timestamps will be sequenced.
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

    # Sequence videos
    sequencer = TimelineSequencer()
    result = await sequencer.sequence_game_videos(game_id, db)

    return SequencingResponse(
        sequenced_count=result.sequenced_count,
        total_duration=result.total_duration,
        gaps=[
            TimelineGapResponse(
                after_video_id=gap.after_video_id,
                before_video_id=gap.before_video_id,
                gap_seconds=gap.gap_seconds,
            )
            for gap in result.gaps
        ],
        overlaps=[
            TimelineOverlapResponse(
                video1_id=overlap.video1_id,
                video2_id=overlap.video2_id,
                overlap_seconds=overlap.overlap_seconds,
            )
            for overlap in result.overlaps
        ],
        warnings=result.warnings,
    )


@router.get("", response_model=TimelineSummaryResponse)
async def get_timeline_summary(
    game_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TimelineSummaryResponse:
    """Get a summary of the game timeline.

    Returns information about:
    - Number of videos (total, sequenced, unsequenced)
    - Total game duration
    - Whether there are gaps or overlaps
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

    # Get timeline summary
    sequencer = TimelineSequencer()
    summary = await sequencer.get_timeline_summary(game_id, db)

    return TimelineSummaryResponse(**summary)
