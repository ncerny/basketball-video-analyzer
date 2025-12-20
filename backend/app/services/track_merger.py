"""Post-processing track merger service.

Merges fragmented ByteTrack tracking IDs that likely belong to the same player.
This compensates for ByteTrack's IOU-only matching which fails when players
move more than their bounding box width between detection gaps.
"""

import logging
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import PlayerDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackMergerConfig:
    """Configuration for track merging."""

    # Maximum frames between end of track A and start of track B to consider merging
    max_temporal_gap_frames: int = 30  # ~1 second at 30fps

    # Maximum center distance (pixels) between tracks to consider merging
    max_spatial_distance: float = 300.0  # Reasonable player movement in 1 second

    # Minimum bbox size similarity ratio (0-1) for merge candidates
    min_size_similarity: float = 0.6

    # Minimum track length to be considered as "anchor" (longer tracks absorb shorter ones)
    min_anchor_length: int = 3


@dataclass
class TrackInfo:
    """Information about a single track."""

    tracking_id: int
    detection_count: int
    first_frame: int
    last_frame: int
    # Position at first detection
    first_x: float
    first_y: float
    first_width: float
    first_height: float
    # Position at last detection
    last_x: float
    last_y: float
    last_width: float
    last_height: float
    avg_confidence: float


@dataclass
class MergeCandidate:
    """A pair of tracks that could potentially be merged."""

    source_track: TrackInfo  # Track to be absorbed (typically shorter/orphan)
    target_track: TrackInfo  # Track to absorb into (typically longer/anchor)
    temporal_gap: int  # Frames between source end and target start
    spatial_distance: float  # Center distance between endpoints
    size_similarity: float  # Bbox size similarity ratio


@dataclass
class TrackMergerResult:
    """Result from running track merger."""

    video_id: int
    original_track_count: int
    merged_track_count: int
    merges_performed: int
    error: str | None = None


# Progress callback type: (current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class TrackMerger:
    """Merges fragmented tracks that likely belong to the same player.

    Uses spatial proximity, temporal gaps, and bbox size similarity to identify
    track fragments that should be merged. Compensates for ByteTrack's IOU-only
    matching limitations.
    """

    def __init__(
        self,
        db: AsyncSession,
        config: TrackMergerConfig | None = None,
    ) -> None:
        """Initialize the track merger.

        Args:
            db: Async database session.
            config: Merger configuration (uses defaults if None).
        """
        self._db = db
        self._config = config or TrackMergerConfig()

    async def merge_tracks(
        self,
        video_id: int,
        progress_callback: ProgressCallback | None = None,
    ) -> TrackMergerResult:
        """Merge fragmented tracks for a video.

        Args:
            video_id: ID of video to process.
            progress_callback: Optional callback for progress updates.

        Returns:
            TrackMergerResult with merge statistics.
        """

        def report_progress(current: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(current, total, message)

        try:
            report_progress(0, 100, "Analyzing tracks...")

            # Step 1: Get all track info
            tracks = await self._get_track_info(video_id)
            original_count = len(tracks)

            if original_count == 0:
                return TrackMergerResult(
                    video_id=video_id,
                    original_track_count=0,
                    merged_track_count=0,
                    merges_performed=0,
                )

            logger.info(f"Video {video_id}: Found {original_count} tracks to analyze")
            report_progress(20, 100, f"Found {original_count} tracks")

            # Step 2: Find merge candidates
            candidates = self._find_merge_candidates(tracks)
            logger.info(f"Video {video_id}: Found {len(candidates)} merge candidates")
            report_progress(40, 100, f"Found {len(candidates)} merge candidates")

            # Step 3: Build merge plan (handle transitive merges)
            merge_plan = self._build_merge_plan(candidates, tracks)
            report_progress(60, 100, f"Planned {len(merge_plan)} merges")

            # Step 4: Execute merges
            merges_performed = 0
            for source_id, target_id in merge_plan.items():
                await self._execute_merge(video_id, source_id, target_id)
                merges_performed += 1

            await self._db.commit()
            report_progress(90, 100, f"Executed {merges_performed} merges")

            # Step 5: Count final tracks
            final_tracks = await self._get_track_info(video_id)
            merged_count = len(final_tracks)

            logger.info(
                f"Video {video_id}: Merged {original_count} tracks down to {merged_count} "
                f"({merges_performed} merges performed)"
            )
            report_progress(100, 100, "Complete")

            return TrackMergerResult(
                video_id=video_id,
                original_track_count=original_count,
                merged_track_count=merged_count,
                merges_performed=merges_performed,
            )

        except Exception as e:
            logger.error(f"Track merge failed for video {video_id}: {e}")
            return TrackMergerResult(
                video_id=video_id,
                original_track_count=0,
                merged_track_count=0,
                merges_performed=0,
                error=str(e),
            )

    async def _get_track_info(self, video_id: int) -> list[TrackInfo]:
        """Get aggregated info for all tracks in a video."""
        # Use raw SQL for complex aggregation
        query = text("""
            WITH track_bounds AS (
                SELECT 
                    tracking_id,
                    COUNT(*) as detection_count,
                    MIN(frame_number) as first_frame,
                    MAX(frame_number) as last_frame,
                    AVG(confidence_score) as avg_confidence
                FROM player_detections
                WHERE video_id = :video_id
                GROUP BY tracking_id
            ),
            first_detections AS (
                SELECT DISTINCT
                    p.tracking_id,
                    FIRST_VALUE(p.bbox_x) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number) as first_x,
                    FIRST_VALUE(p.bbox_y) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number) as first_y,
                    FIRST_VALUE(p.bbox_width) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number) as first_width,
                    FIRST_VALUE(p.bbox_height) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number) as first_height
                FROM player_detections p
                WHERE p.video_id = :video_id
            ),
            last_detections AS (
                SELECT DISTINCT
                    p.tracking_id,
                    FIRST_VALUE(p.bbox_x) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number DESC) as last_x,
                    FIRST_VALUE(p.bbox_y) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number DESC) as last_y,
                    FIRST_VALUE(p.bbox_width) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number DESC) as last_width,
                    FIRST_VALUE(p.bbox_height) OVER (PARTITION BY p.tracking_id ORDER BY p.frame_number DESC) as last_height
                FROM player_detections p
                WHERE p.video_id = :video_id
            )
            SELECT 
                tb.tracking_id,
                tb.detection_count,
                tb.first_frame,
                tb.last_frame,
                tb.avg_confidence,
                fd.first_x,
                fd.first_y,
                fd.first_width,
                fd.first_height,
                ld.last_x,
                ld.last_y,
                ld.last_width,
                ld.last_height
            FROM track_bounds tb
            JOIN first_detections fd ON tb.tracking_id = fd.tracking_id
            JOIN last_detections ld ON tb.tracking_id = ld.tracking_id
            ORDER BY tb.first_frame
        """)

        result = await self._db.execute(query, {"video_id": video_id})
        rows = result.fetchall()

        return [
            TrackInfo(
                tracking_id=row[0],
                detection_count=row[1],
                first_frame=row[2],
                last_frame=row[3],
                avg_confidence=row[4],
                first_x=row[5],
                first_y=row[6],
                first_width=row[7],
                first_height=row[8],
                last_x=row[9],
                last_y=row[10],
                last_width=row[11],
                last_height=row[12],
            )
            for row in rows
        ]

    def _find_merge_candidates(self, tracks: list[TrackInfo]) -> list[MergeCandidate]:
        """Find pairs of tracks that could potentially be merged."""
        candidates = []

        # Sort tracks by first_frame for efficient pairwise comparison
        sorted_tracks = sorted(tracks, key=lambda t: t.first_frame)

        for i, track_a in enumerate(sorted_tracks):
            for track_b in sorted_tracks[i + 1 :]:
                # Only consider if track_b starts after track_a ends
                if track_b.first_frame <= track_a.last_frame:
                    continue

                # Check temporal gap
                temporal_gap = track_b.first_frame - track_a.last_frame
                if temporal_gap > self._config.max_temporal_gap_frames:
                    # Since tracks are sorted, no need to check further
                    break

                # Check spatial distance (center-to-center from track_a end to track_b start)
                center_a_x = track_a.last_x + track_a.last_width / 2
                center_a_y = track_a.last_y + track_a.last_height / 2
                center_b_x = track_b.first_x + track_b.first_width / 2
                center_b_y = track_b.first_y + track_b.first_height / 2

                spatial_distance = (
                    (center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2
                ) ** 0.5

                if spatial_distance > self._config.max_spatial_distance:
                    continue

                # Check size similarity
                area_a = track_a.last_width * track_a.last_height
                area_b = track_b.first_width * track_b.first_height
                size_similarity = (
                    min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0
                )

                if size_similarity < self._config.min_size_similarity:
                    continue

                # Determine which track should absorb the other
                # Prefer longer tracks as anchors
                if track_a.detection_count >= track_b.detection_count:
                    source, target = track_b, track_a
                else:
                    source, target = track_a, track_b

                candidates.append(
                    MergeCandidate(
                        source_track=source,
                        target_track=target,
                        temporal_gap=temporal_gap,
                        spatial_distance=spatial_distance,
                        size_similarity=size_similarity,
                    )
                )

        return candidates

    def _build_merge_plan(
        self,
        candidates: list[MergeCandidate],
        tracks: list[TrackInfo],
    ) -> dict[int, int]:
        """Build a merge plan handling transitive merges.

        If A→B and B→C, we want A→C and B→C.

        Returns:
            Dict mapping source_tracking_id → target_tracking_id
        """
        # Sort candidates by confidence (prefer merging high-confidence matches)
        # Use spatial distance as primary sort (closer = better match)
        sorted_candidates = sorted(candidates, key=lambda c: c.spatial_distance)

        # Track which IDs have been absorbed
        absorbed: set[int] = set()
        # Map source → ultimate target
        merge_map: dict[int, int] = {}

        for candidate in sorted_candidates:
            source_id = candidate.source_track.tracking_id
            target_id = candidate.target_track.tracking_id

            # Skip if source already absorbed
            if source_id in absorbed:
                continue

            # Follow target chain to find ultimate target
            ultimate_target = target_id
            while ultimate_target in merge_map:
                ultimate_target = merge_map[ultimate_target]

            # Don't merge if target was absorbed into something else
            if target_id in absorbed and target_id != ultimate_target:
                continue

            # Record the merge
            merge_map[source_id] = ultimate_target
            absorbed.add(source_id)

            logger.debug(
                f"Merge plan: track {source_id} → {ultimate_target} "
                f"(gap={candidate.temporal_gap}, dist={candidate.spatial_distance:.1f})"
            )

        return merge_map

    async def _execute_merge(
        self,
        video_id: int,
        source_tracking_id: int,
        target_tracking_id: int,
    ) -> None:
        """Update all detections from source track to use target track ID."""
        stmt = (
            update(PlayerDetection)
            .where(PlayerDetection.video_id == video_id)
            .where(PlayerDetection.tracking_id == source_tracking_id)
            .values(tracking_id=target_tracking_id)
        )
        await self._db.execute(stmt)

        logger.debug(f"Merged track {source_tracking_id} into {target_tracking_id}")
