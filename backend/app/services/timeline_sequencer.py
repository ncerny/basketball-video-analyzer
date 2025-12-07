"""Game timeline sequencing service."""

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video as VideoModel


@dataclass
class TimelineGap:
    """Represents a gap between videos in the timeline."""

    after_video_id: int
    before_video_id: int
    gap_seconds: float


@dataclass
class TimelineOverlap:
    """Represents an overlap between videos in the timeline."""

    video1_id: int
    video2_id: int
    overlap_seconds: float


@dataclass
class SequencingResult:
    """Result of timeline sequencing operation."""

    sequenced_count: int
    total_duration: float
    gaps: list[TimelineGap]
    overlaps: list[TimelineOverlap]
    warnings: list[str]


class TimelineSequencer:
    """Service for sequencing game videos in chronological order."""

    async def sequence_game_videos(
        self, game_id: int, db: AsyncSession
    ) -> SequencingResult:
        """Sequence all videos for a game based on recorded_at timestamps.

        This method:
        1. Fetches all videos for the game that have recorded_at timestamps
        2. Sorts them chronologically
        3. Assigns sequence_order (0-indexed)
        4. Calculates game_time_offset for each video
        5. Detects gaps and overlaps between videos

        Args:
            game_id: The game ID to sequence videos for
            db: Database session

        Returns:
            SequencingResult with statistics and detected issues
        """
        # Fetch videos with recorded_at timestamps
        stmt = (
            select(VideoModel)
            .where(VideoModel.game_id == game_id, VideoModel.recorded_at.isnot(None))
            .order_by(VideoModel.recorded_at)
        )
        result = await db.execute(stmt)
        videos = list(result.scalars().all())

        if not videos:
            return SequencingResult(
                sequenced_count=0,
                total_duration=0.0,
                gaps=[],
                overlaps=[],
                warnings=["No videos with recorded_at timestamps found"],
            )

        # Calculate sequence order and game_time_offset
        gaps: list[TimelineGap] = []
        overlaps: list[TimelineOverlap] = []
        warnings: list[str] = []
        current_game_time = 0.0

        for idx, video in enumerate(videos):
            video.sequence_order = idx

            if idx == 0:
                # First video starts at game time 0
                video.game_time_offset = 0.0
            else:
                prev_video = videos[idx - 1]
                time_gap = self._calculate_time_gap(prev_video, video)

                if time_gap > 0:
                    # Gap detected - videos don't overlap
                    video.game_time_offset = current_game_time + time_gap
                    gaps.append(
                        TimelineGap(
                            after_video_id=prev_video.id,
                            before_video_id=video.id,
                            gap_seconds=time_gap,
                        )
                    )
                elif time_gap < 0:
                    # Overlap detected - videos recorded simultaneously
                    overlap_seconds = abs(time_gap)
                    video.game_time_offset = current_game_time - overlap_seconds
                    overlaps.append(
                        TimelineOverlap(
                            video1_id=prev_video.id,
                            video2_id=video.id,
                            overlap_seconds=overlap_seconds,
                        )
                    )
                else:
                    # Perfect continuity
                    video.game_time_offset = current_game_time

            current_game_time = video.game_time_offset + video.duration_seconds

        # Commit changes
        await db.commit()

        # Refresh all videos to get updated values
        for video in videos:
            await db.refresh(video)

        return SequencingResult(
            sequenced_count=len(videos),
            total_duration=current_game_time,
            gaps=gaps,
            overlaps=overlaps,
            warnings=warnings,
        )

    def _calculate_time_gap(self, prev_video: VideoModel, current_video: VideoModel) -> float:
        """Calculate the time gap between two videos.

        Positive value = gap (videos are separated)
        Negative value = overlap (videos overlap in time)
        Zero = perfect continuity

        Args:
            prev_video: The earlier video
            current_video: The later video

        Returns:
            Time gap in seconds (positive for gap, negative for overlap)
        """
        if not prev_video.recorded_at or not current_video.recorded_at:
            # If timestamps missing, assume continuity
            return 0.0

        # Calculate when previous video ends
        prev_end_time = prev_video.recorded_at + dt.timedelta(seconds=prev_video.duration_seconds)

        # Calculate gap (positive) or overlap (negative)
        gap = (current_video.recorded_at - prev_end_time).total_seconds()

        return gap

    async def get_timeline_summary(self, game_id: int, db: AsyncSession) -> dict[str, Any]:
        """Get a summary of the game timeline.

        Args:
            game_id: The game ID
            db: Database session

        Returns:
            Dictionary with timeline information
        """
        stmt = (
            select(VideoModel)
            .where(VideoModel.game_id == game_id)
            .order_by(VideoModel.sequence_order.nullsfirst(), VideoModel.recorded_at.nullsfirst())
        )
        result = await db.execute(stmt)
        videos = list(result.scalars().all())

        if not videos:
            return {
                "game_id": game_id,
                "video_count": 0,
                "total_duration": 0.0,
                "sequenced_count": 0,
                "unsequenced_count": 0,
                "has_gaps": False,
                "has_overlaps": False,
            }

        sequenced_videos = [v for v in videos if v.sequence_order is not None]
        unsequenced_videos = [v for v in videos if v.sequence_order is None]

        # Calculate total duration (only for sequenced videos with game_time_offset)
        total_duration = 0.0
        if sequenced_videos:
            last_video = max(
                (v for v in sequenced_videos if v.game_time_offset is not None),
                key=lambda v: v.game_time_offset,
                default=None,
            )
            if last_video:
                total_duration = last_video.game_time_offset + last_video.duration_seconds

        # Detect gaps and overlaps
        has_gaps = False
        has_overlaps = False
        for i in range(len(sequenced_videos) - 1):
            if sequenced_videos[i].recorded_at and sequenced_videos[i + 1].recorded_at:
                gap = self._calculate_time_gap(sequenced_videos[i], sequenced_videos[i + 1])
                if gap > 0:
                    has_gaps = True
                elif gap < 0:
                    has_overlaps = True

        return {
            "game_id": game_id,
            "video_count": len(videos),
            "total_duration": total_duration,
            "sequenced_count": len(sequenced_videos),
            "unsequenced_count": len(unsequenced_videos),
            "has_gaps": has_gaps,
            "has_overlaps": has_overlaps,
        }
