"""Post-processing track merger service.

Merges fragmented ByteTrack tracking IDs that likely belong to the same player.
This compensates for ByteTrack's IOU-only matching which fails when players
move more than their bounding box width between detection gaps.

Includes jersey number-based merging to consolidate tracks with the same
detected jersey number.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber

logger = logging.getLogger(__name__)


@dataclass
class TrackMergerConfig:
    # Maximum frames between end of track A and start of track B to consider merging
    max_temporal_gap_frames: int = 30  # ~1 second at 30fps

    # Maximum center distance (pixels) between tracks to consider merging
    max_spatial_distance: float = 300.0  # Reasonable player movement in 1 second

    # Minimum bbox size similarity ratio (0-1) for merge candidates
    min_size_similarity: float = 0.6

    # Minimum track length to be considered as "anchor" (longer tracks absorb shorter ones)
    min_anchor_length: int = 3

    # Jersey-based merging settings
    enable_jersey_merge: bool = True
    min_jersey_confidence: float = 0.6  # Minimum confidence to use jersey for merging
    min_jersey_readings: int = 2  # Minimum OCR readings to trust jersey number


@dataclass
class TrackInfo:
    tracking_id: int
    detection_count: int
    first_frame: int
    last_frame: int
    first_x: float
    first_y: float
    first_width: float
    first_height: float
    last_x: float
    last_y: float
    last_width: float
    last_height: float
    avg_confidence: float
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    jersey_readings: int = 0


@dataclass
class MergeCandidate:
    source_track: TrackInfo
    target_track: TrackInfo
    temporal_gap: int
    spatial_distance: float
    size_similarity: float
    merge_reason: str = "spatial"


@dataclass
class TrackMergerResult:
    video_id: int
    original_track_count: int
    merged_track_count: int
    merges_performed: int
    spatial_merges: int = 0
    jersey_merges: int = 0
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
        def report_progress(current: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(current, total, message)

        try:
            report_progress(0, 100, "Analyzing tracks...")

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
            report_progress(10, 100, f"Found {original_count} tracks")

            if self._config.enable_jersey_merge:
                await self._enrich_tracks_with_jersey_data(video_id, tracks)
                jersey_tracks = sum(1 for t in tracks if t.jersey_number is not None)
                logger.info(f"Video {video_id}: {jersey_tracks} tracks have jersey numbers")
                report_progress(20, 100, f"{jersey_tracks} tracks have jersey numbers")

            spatial_candidates = self._find_spatial_merge_candidates(tracks)
            logger.info(
                f"Video {video_id}: Found {len(spatial_candidates)} spatial merge candidates"
            )
            report_progress(30, 100, f"Found {len(spatial_candidates)} spatial candidates")

            jersey_candidates = []
            if self._config.enable_jersey_merge:
                jersey_candidates = self._find_jersey_merge_candidates(tracks)
                logger.info(
                    f"Video {video_id}: Found {len(jersey_candidates)} jersey merge candidates"
                )
                report_progress(40, 100, f"Found {len(jersey_candidates)} jersey candidates")

            all_candidates = spatial_candidates + jersey_candidates
            merge_plan = self._build_merge_plan(all_candidates, tracks)
            report_progress(50, 100, f"Planned {len(merge_plan)} merges")

            spatial_merges = 0
            jersey_merges = 0
            for source_id, (target_id, reason) in merge_plan.items():
                await self._execute_merge(video_id, source_id, target_id)
                if reason == "jersey":
                    jersey_merges += 1
                else:
                    spatial_merges += 1

            await self._db.commit()
            merges_performed = spatial_merges + jersey_merges
            report_progress(80, 100, f"Executed {merges_performed} merges")

            final_tracks = await self._get_track_info(video_id)
            merged_count = len(final_tracks)

            logger.info(
                f"Video {video_id}: Merged {original_count} → {merged_count} tracks "
                f"(spatial={spatial_merges}, jersey={jersey_merges})"
            )
            report_progress(100, 100, "Complete")

            return TrackMergerResult(
                video_id=video_id,
                original_track_count=original_count,
                merged_track_count=merged_count,
                merges_performed=merges_performed,
                spatial_merges=spatial_merges,
                jersey_merges=jersey_merges,
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

    async def _enrich_tracks_with_jersey_data(self, video_id: int, tracks: list[TrackInfo]) -> None:
        query = text("""
            SELECT 
                tracking_id,
                parsed_number,
                COUNT(*) as reading_count,
                SUM(CASE WHEN is_valid THEN confidence ELSE 0 END) as total_conf,
                SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as valid_count
            FROM jersey_numbers
            WHERE video_id = :video_id AND is_valid = 1 AND parsed_number IS NOT NULL
            GROUP BY tracking_id, parsed_number
        """)
        result = await self._db.execute(query, {"video_id": video_id})
        rows = result.fetchall()

        track_jerseys: dict[int, dict] = {}
        for row in rows:
            tracking_id, parsed_number, reading_count, total_conf, valid_count = row
            if tracking_id not in track_jerseys:
                track_jerseys[tracking_id] = {"numbers": {}, "total_readings": 0}
            track_jerseys[tracking_id]["numbers"][parsed_number] = {
                "count": reading_count,
                "conf": total_conf,
            }
            track_jerseys[tracking_id]["total_readings"] += reading_count

        for track in tracks:
            if track.tracking_id in track_jerseys:
                jersey_data = track_jerseys[track.tracking_id]
                numbers = jersey_data["numbers"]
                if numbers:
                    best_number = max(numbers.keys(), key=lambda n: numbers[n]["conf"])
                    best_data = numbers[best_number]
                    total_conf = sum(d["conf"] for d in numbers.values())
                    confidence = best_data["conf"] / total_conf if total_conf > 0 else 0

                    if (
                        best_data["count"] >= self._config.min_jersey_readings
                        and confidence >= self._config.min_jersey_confidence
                    ):
                        track.jersey_number = best_number
                        track.jersey_confidence = confidence
                        track.jersey_readings = best_data["count"]

    def _find_spatial_merge_candidates(self, tracks: list[TrackInfo]) -> list[MergeCandidate]:
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
                        merge_reason="spatial",
                    )
                )

        return candidates

    def _find_jersey_merge_candidates(self, tracks: list[TrackInfo]) -> list[MergeCandidate]:
        candidates = []

        tracks_with_jersey = [t for t in tracks if t.jersey_number is not None]

        jersey_groups: dict[int, list[TrackInfo]] = {}
        for track in tracks_with_jersey:
            jersey = track.jersey_number
            if jersey not in jersey_groups:
                jersey_groups[jersey] = []
            jersey_groups[jersey].append(track)

        for jersey_number, group in jersey_groups.items():
            if len(group) < 2:
                continue

            sorted_group = sorted(group, key=lambda t: (-t.detection_count, -t.jersey_confidence))
            anchor = sorted_group[0]

            for track in sorted_group[1:]:
                if track.first_frame > anchor.last_frame or track.last_frame < anchor.first_frame:
                    candidates.append(
                        MergeCandidate(
                            source_track=track,
                            target_track=anchor,
                            temporal_gap=abs(track.first_frame - anchor.last_frame),
                            spatial_distance=0.0,
                            size_similarity=1.0,
                            merge_reason="jersey",
                        )
                    )
                    logger.debug(
                        f"Jersey merge candidate: track {track.tracking_id} → {anchor.tracking_id} "
                        f"(jersey #{jersey_number})"
                    )

        return candidates

    def _build_merge_plan(
        self,
        candidates: list[MergeCandidate],
        tracks: list[TrackInfo],
    ) -> dict[int, tuple[int, str]]:
        def sort_key(c: MergeCandidate) -> tuple:
            if c.merge_reason == "jersey":
                return (0, -c.source_track.jersey_confidence)
            return (1, c.spatial_distance)

        sorted_candidates = sorted(candidates, key=sort_key)

        absorbed: set[int] = set()
        merge_map: dict[int, tuple[int, str]] = {}

        for candidate in sorted_candidates:
            source_id = candidate.source_track.tracking_id
            target_id = candidate.target_track.tracking_id

            if source_id in absorbed:
                continue

            ultimate_target = target_id
            while ultimate_target in merge_map:
                ultimate_target = merge_map[ultimate_target][0]

            if target_id in absorbed and target_id != ultimate_target:
                continue

            merge_map[source_id] = (ultimate_target, candidate.merge_reason)
            absorbed.add(source_id)

            logger.debug(
                f"Merge plan: track {source_id} → {ultimate_target} "
                f"(reason={candidate.merge_reason}, gap={candidate.temporal_gap})"
            )

        return merge_map

    async def _execute_merge(
        self,
        video_id: int,
        source_tracking_id: int,
        target_tracking_id: int,
    ) -> None:
        detection_stmt = (
            update(PlayerDetection)
            .where(PlayerDetection.video_id == video_id)
            .where(PlayerDetection.tracking_id == source_tracking_id)
            .values(tracking_id=target_tracking_id)
        )
        await self._db.execute(detection_stmt)

        jersey_stmt = (
            update(JerseyNumber)
            .where(JerseyNumber.video_id == video_id)
            .where(JerseyNumber.tracking_id == source_tracking_id)
            .values(tracking_id=target_tracking_id)
        )
        await self._db.execute(jersey_stmt)

        logger.debug(f"Merged track {source_tracking_id} into {target_tracking_id}")
