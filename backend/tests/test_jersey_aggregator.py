from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.jersey_number import JerseyNumber
from app.services.jersey_aggregator import JerseyAggregator, aggregate_jersey_numbers


def make_reading(
    tracking_id: int, parsed_number: int | None, confidence: float, is_valid: bool
) -> MagicMock:
    reading = MagicMock(spec=JerseyNumber)
    reading.tracking_id = tracking_id
    reading.parsed_number = parsed_number
    reading.confidence = confidence
    reading.is_valid = is_valid
    return reading


class TestJerseyAggregator:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=AsyncSession)
        db.execute = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_empty_video(self, mock_db):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        assert result.video_id == 1
        assert result.total_tracks == 0
        assert result.tracks_with_numbers == 0
        assert result.aggregated == []

    @pytest.mark.asyncio
    async def test_single_track_consistent_readings(self, mock_db):
        readings = [
            make_reading(1, 23, 0.9, True),
            make_reading(1, 23, 0.85, True),
            make_reading(1, 23, 0.88, True),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        assert result.total_tracks == 1
        assert result.tracks_with_numbers == 1
        assert len(result.aggregated) == 1

        agg = result.aggregated[0]
        assert agg.tracking_id == 1
        assert agg.jersey_number == 23
        assert agg.confidence == 1.0
        assert agg.valid_readings == 3
        assert agg.has_conflict is False
        assert agg.all_numbers == [23]

    @pytest.mark.asyncio
    async def test_track_with_conflicting_readings(self, mock_db):
        readings = [
            make_reading(1, 23, 0.9, True),
            make_reading(1, 23, 0.85, True),
            make_reading(1, 45, 0.8, True),
            make_reading(1, 45, 0.75, True),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        agg = result.aggregated[0]
        assert agg.jersey_number == 23
        assert agg.has_conflict is True
        assert set(agg.all_numbers) == {23, 45}

    @pytest.mark.asyncio
    async def test_track_below_min_readings(self, mock_db):
        readings = [
            make_reading(1, 23, 0.9, True),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        agg = result.aggregated[0]
        assert agg.jersey_number is None
        assert agg.valid_readings == 1
        assert agg.all_numbers == [23]

    @pytest.mark.asyncio
    async def test_track_with_invalid_readings_only(self, mock_db):
        readings = [
            make_reading(1, None, 0.3, False),
            make_reading(1, None, 0.2, False),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        agg = result.aggregated[0]
        assert agg.jersey_number is None
        assert agg.valid_readings == 0
        assert agg.total_readings == 2

    @pytest.mark.asyncio
    async def test_multiple_tracks(self, mock_db):
        readings = [
            make_reading(1, 23, 0.9, True),
            make_reading(1, 23, 0.85, True),
            make_reading(2, 7, 0.88, True),
            make_reading(2, 7, 0.9, True),
            make_reading(3, None, 0.3, False),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        result = await aggregate_jersey_numbers(mock_db, video_id=1)

        assert result.total_tracks == 3
        assert result.tracks_with_numbers == 2

        track1 = next(a for a in result.aggregated if a.tracking_id == 1)
        track2 = next(a for a in result.aggregated if a.tracking_id == 2)
        track3 = next(a for a in result.aggregated if a.tracking_id == 3)

        assert track1.jersey_number == 23
        assert track2.jersey_number == 7
        assert track3.jersey_number is None

    @pytest.mark.asyncio
    async def test_weighted_voting(self, mock_db):
        readings = [
            make_reading(1, 23, 0.95, True),
            make_reading(1, 45, 0.5, True),
            make_reading(1, 45, 0.5, True),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = readings
        mock_db.execute.return_value = mock_result

        aggregator = JerseyAggregator(mock_db, min_readings=2)
        result = await aggregator.aggregate_for_video(video_id=1)

        agg = result.aggregated[0]
        assert agg.jersey_number == 45
