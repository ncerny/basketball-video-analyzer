from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber
from app.models.video import Video
from app.models.game import Game
from app.services.identity_switch_detector import (
    IdentitySwitchConfig,
    IdentitySwitchDetector,
    JerseyWindow,
)


@pytest.fixture
async def game_with_video(db_session: AsyncSession) -> tuple[Game, Video]:
    game = Game(
        name="Test Game",
        date=date(2025, 1, 1),
        home_team="Home",
        away_team="Away",
    )
    db_session.add(game)
    await db_session.flush()

    video = Video(
        game_id=game.id,
        file_path="test.mp4",
        duration_seconds=60.0,
        fps=30.0,
        resolution="1920x1080",
    )
    db_session.add(video)
    await db_session.flush()

    return game, video


@pytest.fixture
async def track_with_identity_switch(
    db_session: AsyncSession, game_with_video: tuple[Game, Video]
) -> int:
    _, video = game_with_video

    for frame in range(0, 900, 3):
        detection = PlayerDetection(
            video_id=video.id,
            frame_number=frame,
            tracking_id=1,
            bbox_x=100.0,
            bbox_y=100.0,
            bbox_width=50.0,
            bbox_height=150.0,
            confidence_score=0.9,
        )
        db_session.add(detection)

    for frame in range(0, 400, 30):
        jersey = JerseyNumber(
            video_id=video.id,
            frame_number=frame,
            tracking_id=1,
            raw_ocr_output="23",
            parsed_number=23,
            confidence=0.9,
            is_valid=True,
        )
        db_session.add(jersey)

    for frame in range(500, 900, 30):
        jersey = JerseyNumber(
            video_id=video.id,
            frame_number=frame,
            tracking_id=1,
            raw_ocr_output="45",
            parsed_number=45,
            confidence=0.9,
            is_valid=True,
        )
        db_session.add(jersey)

    await db_session.commit()
    return video.id


@pytest.fixture
async def track_without_identity_switch(
    db_session: AsyncSession, game_with_video: tuple[Game, Video]
) -> int:
    _, video = game_with_video

    for frame in range(0, 900, 3):
        detection = PlayerDetection(
            video_id=video.id,
            frame_number=frame,
            tracking_id=1,
            bbox_x=100.0,
            bbox_y=100.0,
            bbox_width=50.0,
            bbox_height=150.0,
            confidence_score=0.9,
        )
        db_session.add(detection)

    for frame in range(0, 900, 30):
        jersey = JerseyNumber(
            video_id=video.id,
            frame_number=frame,
            tracking_id=1,
            raw_ocr_output="23",
            parsed_number=23,
            confidence=0.9,
            is_valid=True,
        )
        db_session.add(jersey)

    await db_session.commit()
    return video.id


class TestIdentitySwitchDetector:
    async def test_detects_identity_switch(
        self, db_session: AsyncSession, track_with_identity_switch: int
    ):
        video_id = track_with_identity_switch

        config = IdentitySwitchConfig(
            window_size_frames=150,
            min_readings_per_window=3,
            switch_threshold=0.7,
        )
        detector = IdentitySwitchDetector(db_session, config)

        result = await detector.detect_and_split(video_id)

        assert result.error is None
        assert result.tracks_analyzed == 1
        assert result.switches_detected == 1
        assert result.tracks_split == 1

    async def test_no_split_when_consistent_jersey(
        self, db_session: AsyncSession, track_without_identity_switch: int
    ):
        video_id = track_without_identity_switch

        config = IdentitySwitchConfig(
            window_size_frames=150,
            min_readings_per_window=3,
            switch_threshold=0.7,
        )
        detector = IdentitySwitchDetector(db_session, config)

        result = await detector.detect_and_split(video_id)

        assert result.error is None
        assert result.tracks_analyzed == 1
        assert result.switches_detected == 0
        assert result.tracks_split == 0

    async def test_empty_video(self, db_session: AsyncSession, game_with_video: tuple[Game, Video]):
        _, video = game_with_video

        detector = IdentitySwitchDetector(db_session)
        result = await detector.detect_and_split(video.id)

        assert result.error is None
        assert result.tracks_analyzed == 0
        assert result.switches_detected == 0
        assert result.tracks_split == 0

    async def test_track_split_creates_new_tracking_id(
        self, db_session: AsyncSession, track_with_identity_switch: int
    ):
        video_id = track_with_identity_switch

        config = IdentitySwitchConfig(
            window_size_frames=150,
            min_readings_per_window=3,
            switch_threshold=0.7,
        )
        detector = IdentitySwitchDetector(db_session, config)

        await detector.detect_and_split(video_id)

        from sqlalchemy import text

        result = await db_session.execute(
            text("SELECT DISTINCT tracking_id FROM player_detections WHERE video_id = :vid"),
            {"vid": video_id},
        )
        tracking_ids = {row[0] for row in result.fetchall()}

        assert len(tracking_ids) == 2
        assert 1 in tracking_ids
        assert 2 in tracking_ids


class TestJerseyWindowCreation:
    def test_create_jersey_windows(self):
        config = IdentitySwitchConfig(
            window_size_frames=150,
            min_readings_per_window=3,
            min_confidence=0.5,
        )
        detector = IdentitySwitchDetector.__new__(IdentitySwitchDetector)
        detector._config = config

        readings = [
            (0, 23, 0.9),
            (30, 23, 0.8),
            (60, 23, 0.9),
            (90, 23, 0.85),
            (150, 45, 0.9),
            (180, 45, 0.85),
            (210, 45, 0.9),
        ]

        windows = detector._create_jersey_windows(readings)

        assert len(windows) == 2
        assert windows[0].dominant_number == 23
        assert windows[1].dominant_number == 45

    def test_get_dominant_jersey(self):
        config = IdentitySwitchConfig(min_confidence=0.5)
        detector = IdentitySwitchDetector.__new__(IdentitySwitchDetector)
        detector._config = config

        readings = [
            (0, 23, 0.9),
            (30, 23, 0.8),
            (60, 45, 0.6),
        ]

        number, confidence = detector._get_dominant_jersey(readings)

        assert number == 23
        assert confidence > 0.7
