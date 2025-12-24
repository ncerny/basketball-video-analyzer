"""Identity switch detection and track splitting service.

Detects when a track's jersey number changes mid-video (indicating the tracker
switched from following one player to another) and splits the track accordingly.
"""

import logging
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber

logger = logging.getLogger(__name__)


@dataclass
class IdentitySwitchConfig:
    min_readings_per_window: int = 3
    window_size_frames: int = 150
    min_confidence: float = 0.5
    switch_threshold: float = 0.7


@dataclass
class JerseyWindow:
    start_frame: int
    end_frame: int
    dominant_number: int | None
    readings: int
    confidence: float


@dataclass
class IdentitySwitch:
    tracking_id: int
    switch_frame: int
    from_jersey: int
    to_jersey: int
    confidence: float


@dataclass
class IdentitySwitchResult:
    video_id: int
    tracks_analyzed: int
    switches_detected: int
    tracks_split: int
    error: str | None = None


ProgressCallback = Callable[[int, int, str], None]


class IdentitySwitchDetector:
    def __init__(
        self,
        db: AsyncSession,
        config: IdentitySwitchConfig | None = None,
    ) -> None:
        self._db = db
        self._config = config or IdentitySwitchConfig()

    async def detect_and_split(
        self,
        video_id: int,
        progress_callback: ProgressCallback | None = None,
    ) -> IdentitySwitchResult:
        def report_progress(current: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(current, total, message)

        try:
            report_progress(0, 100, "Analyzing tracks for identity switches...")

            track_jerseys = await self._get_jersey_readings_by_track(video_id)
            tracks_analyzed = len(track_jerseys)

            if tracks_analyzed == 0:
                return IdentitySwitchResult(
                    video_id=video_id,
                    tracks_analyzed=0,
                    switches_detected=0,
                    tracks_split=0,
                )

            logger.info(
                f"Video {video_id}: Analyzing {tracks_analyzed} tracks for identity switches"
            )
            report_progress(10, 100, f"Analyzing {tracks_analyzed} tracks")

            switches = []
            for tracking_id, readings in track_jerseys.items():
                track_switches = self._detect_switches_in_track(tracking_id, readings)
                switches.extend(track_switches)

            switches_detected = len(switches)
            logger.info(f"Video {video_id}: Found {switches_detected} identity switches")
            report_progress(50, 100, f"Found {switches_detected} switches")

            if switches_detected == 0:
                return IdentitySwitchResult(
                    video_id=video_id,
                    tracks_analyzed=tracks_analyzed,
                    switches_detected=0,
                    tracks_split=0,
                )

            max_tracking_id = await self._get_max_tracking_id(video_id)
            next_tracking_id = max_tracking_id + 1

            tracks_split = 0
            for switch in switches:
                await self._split_track_at_frame(
                    video_id, switch.tracking_id, switch.switch_frame, next_tracking_id
                )
                logger.info(
                    f"Split track {switch.tracking_id} at frame {switch.switch_frame} "
                    f"(jersey #{switch.from_jersey} → #{switch.to_jersey}), new track: {next_tracking_id}"
                )
                next_tracking_id += 1
                tracks_split += 1

            await self._db.commit()
            report_progress(100, 100, f"Split {tracks_split} tracks")

            return IdentitySwitchResult(
                video_id=video_id,
                tracks_analyzed=tracks_analyzed,
                switches_detected=switches_detected,
                tracks_split=tracks_split,
            )

        except Exception as e:
            logger.error(f"Identity switch detection failed for video {video_id}: {e}")
            return IdentitySwitchResult(
                video_id=video_id,
                tracks_analyzed=0,
                switches_detected=0,
                tracks_split=0,
                error=str(e),
            )

    async def _get_jersey_readings_by_track(
        self, video_id: int
    ) -> dict[int, list[tuple[int, int, float]]]:
        query = text("""
            SELECT tracking_id, frame_number, parsed_number, confidence
            FROM jersey_numbers
            WHERE video_id = :video_id AND is_valid = 1 AND parsed_number IS NOT NULL
            ORDER BY tracking_id, frame_number
        """)
        result = await self._db.execute(query, {"video_id": video_id})
        rows = result.fetchall()

        track_readings: dict[int, list[tuple[int, int, float]]] = {}
        for row in rows:
            tracking_id, frame_number, parsed_number, confidence = row
            if tracking_id not in track_readings:
                track_readings[tracking_id] = []
            track_readings[tracking_id].append((frame_number, parsed_number, confidence))

        return track_readings

    def _detect_switches_in_track(
        self, tracking_id: int, readings: list[tuple[int, int, float]]
    ) -> list[IdentitySwitch]:
        if len(readings) < self._config.min_readings_per_window * 2:
            return []

        windows = self._create_jersey_windows(readings)

        if len(windows) < 2:
            return []

        switches = []
        for i in range(1, len(windows)):
            prev_window = windows[i - 1]
            curr_window = windows[i]

            if prev_window.dominant_number is None or curr_window.dominant_number is None:
                continue

            if prev_window.dominant_number != curr_window.dominant_number:
                if (
                    prev_window.confidence >= self._config.switch_threshold
                    and curr_window.confidence >= self._config.switch_threshold
                ):
                    switches.append(
                        IdentitySwitch(
                            tracking_id=tracking_id,
                            switch_frame=curr_window.start_frame,
                            from_jersey=prev_window.dominant_number,
                            to_jersey=curr_window.dominant_number,
                            confidence=min(prev_window.confidence, curr_window.confidence),
                        )
                    )
                    logger.debug(
                        f"Track {tracking_id}: identity switch at frame {curr_window.start_frame} "
                        f"(#{prev_window.dominant_number} → #{curr_window.dominant_number})"
                    )

        return switches

    def _create_jersey_windows(self, readings: list[tuple[int, int, float]]) -> list[JerseyWindow]:
        if not readings:
            return []

        first_frame = readings[0][0]
        last_frame = readings[-1][0]
        window_size = self._config.window_size_frames

        windows = []
        current_start = first_frame

        while current_start <= last_frame:
            current_end = current_start + window_size - 1

            window_readings = [
                (frame, number, conf)
                for frame, number, conf in readings
                if current_start <= frame <= current_end
            ]

            if len(window_readings) >= self._config.min_readings_per_window:
                dominant_number, confidence = self._get_dominant_jersey(window_readings)
                windows.append(
                    JerseyWindow(
                        start_frame=current_start,
                        end_frame=min(current_end, last_frame),
                        dominant_number=dominant_number,
                        readings=len(window_readings),
                        confidence=confidence,
                    )
                )

            current_start += window_size

        return windows

    def _get_dominant_jersey(
        self, readings: list[tuple[int, int, float]]
    ) -> tuple[int | None, float]:
        number_scores: dict[int, float] = {}

        for _, number, conf in readings:
            if conf >= self._config.min_confidence:
                number_scores[number] = number_scores.get(number, 0) + conf

        if not number_scores:
            return None, 0.0

        dominant_number = max(number_scores.keys(), key=lambda n: number_scores[n])
        total_score = sum(number_scores.values())
        confidence = number_scores[dominant_number] / total_score if total_score > 0 else 0.0

        return dominant_number, confidence

    async def _get_max_tracking_id(self, video_id: int) -> int:
        query = text("""
            SELECT COALESCE(MAX(tracking_id), 0) FROM player_detections WHERE video_id = :video_id
        """)
        result = await self._db.execute(query, {"video_id": video_id})
        return result.scalar() or 0

    async def _split_track_at_frame(
        self,
        video_id: int,
        tracking_id: int,
        split_frame: int,
        new_tracking_id: int,
    ) -> None:
        detection_stmt = (
            update(PlayerDetection)
            .where(PlayerDetection.video_id == video_id)
            .where(PlayerDetection.tracking_id == tracking_id)
            .where(PlayerDetection.frame_number >= split_frame)
            .values(tracking_id=new_tracking_id)
        )
        await self._db.execute(detection_stmt)

        jersey_stmt = (
            update(JerseyNumber)
            .where(JerseyNumber.video_id == video_id)
            .where(JerseyNumber.tracking_id == tracking_id)
            .where(JerseyNumber.frame_number >= split_frame)
            .values(tracking_id=new_tracking_id)
        )
        await self._db.execute(jersey_stmt)
