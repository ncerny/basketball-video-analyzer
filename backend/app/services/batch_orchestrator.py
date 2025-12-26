import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import settings
from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber
from app.models.processing_batch import BatchStatus, ProcessingBatch
from app.models.video import ProcessingStatus, Video
from app.services.batch_processor import (
    BatchConfig,
    DetectionBatchProcessor,
    DetectionBatchResult,
    OCRBatchProcessor,
    OCRBatchResult,
)
from app.services.frame_extractor import FrameExtractor
from app.services.identity_switch_detector import IdentitySwitchConfig, IdentitySwitchDetector
from app.services.track_merger import TrackMerger, TrackMergerConfig

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class OrchestratorConfig:
    frames_per_batch: int = 30
    sample_interval: int = 3
    confidence_threshold: float = 0.5
    device: str = "cpu"
    enable_tracking: bool = True
    enable_court_detection: bool = True
    enable_jersey_ocr: bool = True
    enable_identity_switch_detection: bool = True
    enable_track_merging: bool = True
    max_seconds: float | None = None
    delete_existing: bool = True


@dataclass
class OrchestratorResult:
    video_id: int
    batches_processed: int
    total_frames: int
    total_detections: int
    total_ocr_results: int
    resumed_from_batch: int | None
    error: str | None = None


class SequentialOrchestrator:
    def __init__(self, db: AsyncSession, config: OrchestratorConfig | None = None) -> None:
        self._db = db
        self._config = config or OrchestratorConfig(
            confidence_threshold=settings.yolo_confidence_threshold,
            device=self._resolve_device(settings.ml_device),
            enable_jersey_ocr=settings.enable_jersey_ocr,
        )
        self._video_storage_path = Path(settings.video_storage_path)

    @staticmethod
    def _resolve_device(device_setting: str) -> str:
        if device_setting == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return device_setting

    async def process_video(
        self,
        video_id: int,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestratorResult:
        def report(pct: int, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, 100, msg)

        try:
            report(0, "[1/4] Setup: Loading video...")
            video = await self._get_video(video_id)
            if not video:
                return OrchestratorResult(
                    video_id=video_id,
                    batches_processed=0,
                    total_frames=0,
                    total_detections=0,
                    total_ocr_results=0,
                    resumed_from_batch=None,
                    error=f"Video not found: {video_id}",
                )

            await self._update_video_status(video, ProcessingStatus.PROCESSING)
            video_path = self._video_storage_path / video.file_path

            if not video_path.exists():
                await self._update_video_status(video, ProcessingStatus.FAILED)
                return OrchestratorResult(
                    video_id=video_id,
                    batches_processed=0,
                    total_frames=0,
                    total_detections=0,
                    total_ocr_results=0,
                    resumed_from_batch=None,
                    error=f"Video file not found: {video_path}",
                )

            resume_batch_index = await self._find_resume_point(video_id)
            has_existing_batches = await self._has_existing_batches(video_id)

            if (
                resume_batch_index is None
                and not has_existing_batches
                and self._config.delete_existing
            ):
                report(3, "[1/4] Setup: Clearing existing data...")
                await self._delete_existing_data(video_id)

            report(5, "[1/4] Setup: Creating batch plan...")
            batches = await self._create_or_extend_batches(video_id, video_path)

            if not batches:
                report(100, "[4/4] Complete: No frames to process")
                await self._update_video_status(video, ProcessingStatus.COMPLETED, processed=True)
                return OrchestratorResult(
                    video_id=video_id,
                    batches_processed=0,
                    total_frames=0,
                    total_detections=0,
                    total_ocr_results=0,
                    resumed_from_batch=resume_batch_index,
                )

            batch_config = BatchConfig(
                sample_interval=self._config.sample_interval,
                confidence_threshold=self._config.confidence_threshold,
                device=self._config.device,
                enable_tracking=self._config.enable_tracking,
                enable_court_detection=self._config.enable_court_detection,
                enable_jersey_ocr=self._config.enable_jersey_ocr,
                ocr_max_workers=settings.ocr_max_workers,
            )

            detection_processor = DetectionBatchProcessor(self._db, batch_config)
            ocr_processor = OCRBatchProcessor(self._db, batch_config)

            if resume_batch_index is not None:
                logger.info(f"Resuming from batch {resume_batch_index}")

            total_detections = 0
            total_ocr_results = 0
            total_frames = 0

            with FrameExtractor(video_path) as extractor:
                metadata = extractor.get_metadata()
                fps = metadata.fps

                pending_batches = [
                    b for b in batches if b.detection_status != BatchStatus.COMPLETED
                ]
                total_batches = len(batches)

                for i, batch in enumerate(pending_batches):
                    batch_pct = (i + 1) / len(pending_batches)
                    overall_pct = 10 + int(batch_pct * 75)

                    report(
                        overall_pct,
                        f"[2/4] Detection: Batch {batch.batch_index + 1}/{total_batches}",
                    )

                    frames, frame_numbers = self._extract_batch_frames(
                        extractor, batch.frame_start, batch.frame_end
                    )

                    if not frames:
                        batch.detection_status = BatchStatus.SKIPPED
                        batch.ocr_status = BatchStatus.SKIPPED
                        await self._db.commit()
                        continue

                    det_result = await detection_processor.process_batch(
                        batch, frames, frame_numbers, video_id, fps
                    )
                    total_detections += det_result.detections_created
                    total_frames += det_result.frames_processed

                    report(overall_pct, f"[3/4] OCR: Batch {batch.batch_index + 1}/{total_batches}")

                    ocr_result = await ocr_processor.process_batch(
                        batch, frames, frame_numbers, video_id
                    )
                    total_ocr_results += ocr_result.ocr_results_created

            if self._config.enable_identity_switch_detection:
                report(87, "[4/5] Identity Switch Detection: Splitting switched tracks...")
                await self._run_identity_switch_detector(video_id)

            if self._config.enable_track_merging:
                report(93, "[5/5] Track Merge: Merging fragmented tracks...")
                await self._run_track_merger(video_id)

            report(100, "[5/5] Complete")
            await self._update_video_status(video, ProcessingStatus.COMPLETED, processed=True)

            return OrchestratorResult(
                video_id=video_id,
                batches_processed=len(pending_batches),
                total_frames=total_frames,
                total_detections=total_detections,
                total_ocr_results=total_ocr_results,
                resumed_from_batch=resume_batch_index,
            )

        except Exception as e:
            logger.exception(f"Orchestrator failed for video {video_id}")
            try:
                video = await self._get_video(video_id)
                if video:
                    await self._update_video_status(video, ProcessingStatus.FAILED)
            except Exception:
                pass
            return OrchestratorResult(
                video_id=video_id,
                batches_processed=0,
                total_frames=0,
                total_detections=0,
                total_ocr_results=0,
                resumed_from_batch=None,
                error=str(e),
            )

    async def _get_video(self, video_id: int) -> Video | None:
        result = await self._db.execute(select(Video).where(Video.id == video_id))
        return result.scalar_one_or_none()

    async def _update_video_status(
        self, video: Video, status: ProcessingStatus, processed: bool | None = None
    ) -> None:
        video.processing_status = status
        if processed is not None:
            video.processed = processed
        await self._db.commit()

    async def _find_resume_point(self, video_id: int) -> int | None:
        result = await self._db.execute(
            select(ProcessingBatch)
            .where(
                ProcessingBatch.video_id == video_id,
                ProcessingBatch.detection_status != BatchStatus.COMPLETED,
            )
            .order_by(ProcessingBatch.batch_index)
            .limit(1)
        )
        incomplete_batch = result.scalar_one_or_none()

        if incomplete_batch:
            return incomplete_batch.batch_index

        return None

    async def _has_existing_batches(self, video_id: int) -> bool:
        result = await self._db.execute(
            select(ProcessingBatch).where(ProcessingBatch.video_id == video_id).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _delete_existing_data(self, video_id: int) -> None:
        await self._db.execute(delete(JerseyNumber).where(JerseyNumber.video_id == video_id))
        await self._db.execute(delete(PlayerDetection).where(PlayerDetection.video_id == video_id))
        await self._db.execute(delete(ProcessingBatch).where(ProcessingBatch.video_id == video_id))
        await self._db.commit()

    async def _create_or_extend_batches(
        self, video_id: int, video_path: Path
    ) -> list[ProcessingBatch]:
        result = await self._db.execute(
            select(ProcessingBatch)
            .where(ProcessingBatch.video_id == video_id)
            .order_by(ProcessingBatch.batch_index)
        )
        existing_batches = list(result.scalars().all())

        with FrameExtractor(video_path) as extractor:
            metadata = extractor.get_metadata()

            end_frame = metadata.total_frames
            if self._config.max_seconds is not None:
                end_frame = min(int(self._config.max_seconds * metadata.fps), metadata.total_frames)

            frames_per_batch = self._config.frames_per_batch * self._config.sample_interval

            if existing_batches:
                last_batch = existing_batches[-1]
                if last_batch.frame_end >= end_frame:
                    return existing_batches

                frame_start = last_batch.frame_end
                batch_index = last_batch.batch_index + 1
                logger.info(f"Extending batches: adding from frame {frame_start} to {end_frame}")
            else:
                frame_start = 0
                batch_index = 0

            new_batches = []
            while frame_start < end_frame:
                frame_end = min(frame_start + frames_per_batch, end_frame)

                batch = ProcessingBatch(
                    video_id=video_id,
                    batch_index=batch_index,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    detection_status=BatchStatus.PENDING,
                    ocr_status=BatchStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                )
                self._db.add(batch)
                new_batches.append(batch)

                frame_start = frame_end
                batch_index += 1

            if new_batches:
                await self._db.commit()
                for batch in new_batches:
                    await self._db.refresh(batch)

            return existing_batches + new_batches

    def _extract_batch_frames(
        self, extractor: FrameExtractor, frame_start: int, frame_end: int
    ) -> tuple[list[np.ndarray], list[int]]:
        frames = []
        frame_numbers = []

        for extracted in extractor.extract_frames_sampled(
            sample_interval=self._config.sample_interval,
            start_frame=frame_start,
            end_frame=frame_end,
        ):
            frames.append(extracted.frame)
            frame_numbers.append(extracted.frame_number)

        return frames, frame_numbers

    async def _run_identity_switch_detector(self, video_id: int) -> None:
        switch_config = IdentitySwitchConfig(
            window_size_frames=settings.identity_switch_window_size_frames,
            min_readings_per_window=settings.identity_switch_min_readings,
            switch_threshold=settings.identity_switch_threshold,
        )
        detector = IdentitySwitchDetector(self._db, switch_config)
        result = await detector.detect_and_split(video_id)
        if result.error:
            logger.warning(f"Identity switch detection failed: {result.error}")
        else:
            logger.info(
                f"Identity switch detection: {result.tracks_analyzed} tracks analyzed, "
                f"{result.switches_detected} switches found, {result.tracks_split} tracks split"
            )

    async def _run_track_merger(self, video_id: int) -> None:
        merger_config = TrackMergerConfig(
            enable_jersey_merge=settings.enable_jersey_merge,
            min_jersey_confidence=settings.min_jersey_confidence,
            min_jersey_readings=settings.min_jersey_readings,
        )
        merger = TrackMerger(self._db, merger_config)
        result = await merger.merge_tracks(video_id)
        if result.error:
            logger.warning(f"Track merge failed: {result.error}")
        else:
            logger.info(
                f"Track merge: {result.original_track_count} → {result.merged_track_count} tracks"
            )
