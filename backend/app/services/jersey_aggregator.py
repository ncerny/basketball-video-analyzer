from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.jersey_number import JerseyNumber


@dataclass
class AggregatedJerseyNumber:
    tracking_id: int
    jersey_number: int | None
    confidence: float
    total_readings: int
    valid_readings: int
    has_conflict: bool
    all_numbers: list[int]


@dataclass
class AggregationResult:
    video_id: int
    aggregated: list[AggregatedJerseyNumber]
    total_tracks: int
    tracks_with_numbers: int


class JerseyAggregator:
    def __init__(self, db: AsyncSession, min_readings: int = 2, conflict_threshold: float = 0.3):
        self._db = db
        self._min_readings = min_readings
        self._conflict_threshold = conflict_threshold

    async def aggregate_for_video(self, video_id: int) -> AggregationResult:
        result = await self._db.execute(
            select(JerseyNumber)
            .where(JerseyNumber.video_id == video_id)
            .order_by(JerseyNumber.tracking_id, JerseyNumber.frame_number)
        )
        readings = result.scalars().all()

        by_track: dict[int, list[JerseyNumber]] = {}
        for reading in readings:
            if reading.tracking_id not in by_track:
                by_track[reading.tracking_id] = []
            by_track[reading.tracking_id].append(reading)

        aggregated: list[AggregatedJerseyNumber] = []
        tracks_with_numbers = 0

        for tracking_id, track_readings in by_track.items():
            agg = self._aggregate_track(tracking_id, track_readings)
            aggregated.append(agg)
            if agg.jersey_number is not None:
                tracks_with_numbers += 1

        aggregated.sort(key=lambda x: x.tracking_id)

        return AggregationResult(
            video_id=video_id,
            aggregated=aggregated,
            total_tracks=len(by_track),
            tracks_with_numbers=tracks_with_numbers,
        )

    def _aggregate_track(
        self, tracking_id: int, readings: list[JerseyNumber]
    ) -> AggregatedJerseyNumber:
        valid_readings = [r for r in readings if r.is_valid and r.parsed_number is not None]

        if not valid_readings:
            return AggregatedJerseyNumber(
                tracking_id=tracking_id,
                jersey_number=None,
                confidence=0.0,
                total_readings=len(readings),
                valid_readings=0,
                has_conflict=False,
                all_numbers=[],
            )

        numbers = [r.parsed_number for r in valid_readings if r.parsed_number is not None]
        confidences = [r.confidence for r in valid_readings]

        weighted_votes: dict[int, float] = {}
        for num, conf in zip(numbers, confidences):
            weighted_votes[num] = weighted_votes.get(num, 0) + conf

        if not weighted_votes:
            return AggregatedJerseyNumber(
                tracking_id=tracking_id,
                jersey_number=None,
                confidence=0.0,
                total_readings=len(readings),
                valid_readings=len(valid_readings),
                has_conflict=False,
                all_numbers=numbers,
            )

        best_number = max(weighted_votes, key=lambda k: weighted_votes[k])
        total_weight = sum(weighted_votes.values())
        best_weight = weighted_votes[best_number]
        confidence = best_weight / total_weight if total_weight > 0 else 0.0

        unique_numbers = set(numbers)
        has_conflict = False
        if len(unique_numbers) > 1:
            counter = Counter(numbers)
            most_common_count = counter.most_common(1)[0][1]
            second_ratio = (len(numbers) - most_common_count) / len(numbers)
            has_conflict = second_ratio >= self._conflict_threshold

        final_number = best_number if len(valid_readings) >= self._min_readings else None
        if final_number is None:
            confidence = 0.0

        return AggregatedJerseyNumber(
            tracking_id=tracking_id,
            jersey_number=final_number,
            confidence=confidence,
            total_readings=len(readings),
            valid_readings=len(valid_readings),
            has_conflict=has_conflict,
            all_numbers=sorted(set(numbers)),
        )


async def aggregate_jersey_numbers(db: AsyncSession, video_id: int) -> AggregationResult:
    aggregator = JerseyAggregator(db)
    return await aggregator.aggregate_for_video(video_id)
