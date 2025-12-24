import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import settings
from app.ml.base import BaseDetector
from app.ml.byte_tracker import PlayerTracker
from app.ml.color_extractor import extract_jersey_color, extract_shoe_color
from app.ml.court_detector import CourtDetector
from app.ml.jersey_ocr import JerseyOCR, OCRConfig, OCRResult
from app.ml.legibility_filter import LegibilityConfig, check_legibility, extract_jersey_crop
from app.ml.norfair_tracker import NorfairTracker
from app.ml.sam2_tracker import SAM2VideoTracker, SAM2TrackerConfig
from app.ml.types import BoundingBox, Detection, FrameDetections
from app.ml.yolo_detector import YOLODetector
from app.models.detection import PlayerDetection
from app.models.jersey_number import JerseyNumber
from app.models.processing_batch import BatchStatus, ProcessingBatch

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    sample_interval: int = 3
    batch_size: int = 8
    confidence_threshold: float = 0.5
    device: str = "cpu"
    enable_tracking: bool = True
    track_activation_threshold: float = 0.25
    tracking_buffer_seconds: float = 5.0
    tracking_iou_threshold: float = 0.35
    enable_court_detection: bool = True
    court_overlap_threshold: float = 0.2
    enable_jersey_ocr: bool = True
    ocr_sample_rate: int = 10
    ocr_max_workers: int = 4


@dataclass
class DetectionBatchResult:
    batch_id: int
    frames_processed: int
    detections_created: int
    persons_detected: int
    balls_detected: int


@dataclass
class OCRBatchResult:
    batch_id: int
    detections_processed: int
    ocr_results_created: int


@dataclass
class _OCRWorkItem:
    crop: np.ndarray
    video_id: int
    frame_number: int
    tracking_id: int


class DetectionBatchProcessor:
    def __init__(self, db: AsyncSession, config: BatchConfig) -> None:
        self._db = db
        self._config = config
        self._detector: BaseDetector | None = None
        self._tracker: PlayerTracker | NorfairTracker | SAM2VideoTracker | None = None
        self._court_detector: CourtDetector | None = None

    def _get_detector(self) -> BaseDetector:
        if self._detector is None:
            if settings.detection_backend == "rfdetr":
                from app.ml.rfdetr_detector import RFDETRDetector

                self._detector = RFDETRDetector(
                    confidence_threshold=self._config.confidence_threshold,
                    device=self._config.device,
                )
            else:
                self._detector = YOLODetector(
                    model_path=settings.yolo_model_name,
                    confidence_threshold=self._config.confidence_threshold,
                    device=self._config.device,
                )
        return self._detector

    def _get_tracker(self, fps: float) -> PlayerTracker | NorfairTracker | SAM2VideoTracker:
        if self._tracker is None:
            buffer_frames = int(self._config.tracking_buffer_seconds * fps)
            if settings.tracking_backend == "sam2":
                config = SAM2TrackerConfig(
                    model_name=settings.sam2_model_name,
                    device=self._config.device,
                    new_object_iou_threshold=settings.sam2_new_object_iou_threshold,
                    lost_track_frames=buffer_frames,
                    max_memory_frames=settings.sam2_max_memory_frames,
                )
                self._tracker = SAM2VideoTracker(config)
            elif settings.tracking_backend == "norfair":
                self._tracker = NorfairTracker(
                    distance_threshold=250.0,
                    hit_counter_max=buffer_frames,
                    initialization_delay=1,
                )
            else:
                self._tracker = PlayerTracker(
                    track_activation_threshold=self._config.track_activation_threshold,
                    lost_track_buffer=buffer_frames,
                    minimum_matching_threshold=self._config.tracking_iou_threshold,
                    frame_rate=int(fps),
                )
        return self._tracker

    def _get_court_detector(self) -> CourtDetector:
        if self._court_detector is None:
            self._court_detector = CourtDetector()
        return self._court_detector

    def reset_tracker(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    async def process_batch(
        self,
        batch: ProcessingBatch,
        frames: list[np.ndarray],
        frame_numbers: list[int],
        video_id: int,
        fps: float,
    ) -> DetectionBatchResult:
        batch.detection_status = BatchStatus.PROCESSING
        await self._db.commit()

        detector = self._get_detector()
        tracker = (
            self._get_tracker(fps / self._config.sample_interval)
            if self._config.enable_tracking
            else None
        )
        court_detector = self._get_court_detector() if self._config.enable_court_detection else None

        total_detections = 0
        persons_detected = 0
        balls_detected = 0

        batch_start_time = time.time()
        detections_list = await asyncio.to_thread(detector.detect_batch, frames, frame_numbers[0])
        batch_inference_time = time.time() - batch_start_time

        if settings.enable_inference_timing:
            fps_achieved = len(frames) / batch_inference_time
            logger.info(
                f"Detection batch {batch.batch_index}: {len(frames)} frames in "
                f"{batch_inference_time:.2f}s ({fps_achieved:.1f} fps)"
            )

        for i, frame_detections in enumerate(detections_list):
            frame = frames[i]

            person_detections, other_detections = self._filter_persons_only(frame_detections)
            person_detections = self._extract_colors(person_detections, frame)

            if tracker:
                # SAM2 requires frame images for appearance-based tracking
                if isinstance(tracker, SAM2VideoTracker):
                    person_detections = await asyncio.to_thread(
                        tracker.update, person_detections, frame
                    )
                else:
                    person_detections = await asyncio.to_thread(tracker.update, person_detections)

            if court_detector:
                person_detections = await asyncio.to_thread(
                    self._filter_by_court, person_detections, frame, court_detector
                )

            final_detections = FrameDetections(
                frame_number=frame_detections.frame_number,
                detections=person_detections.detections + other_detections,
                frame_width=frame_detections.frame_width,
                frame_height=frame_detections.frame_height,
            )

            stats = await self._store_detections(video_id, final_detections)
            total_detections += stats["total"]
            persons_detected += stats["persons"]
            balls_detected += stats["balls"]

        await self._db.commit()

        batch.detection_status = BatchStatus.COMPLETED
        batch.detection_completed_at = datetime.now(timezone.utc)
        await self._db.commit()

        return DetectionBatchResult(
            batch_id=batch.id,
            frames_processed=len(frames),
            detections_created=total_detections,
            persons_detected=persons_detected,
            balls_detected=balls_detected,
        )

    def _filter_persons_only(
        self, frame_detections: FrameDetections
    ) -> tuple[FrameDetections, list[Detection]]:
        persons = [d for d in frame_detections.detections if d.is_person]
        others = [d for d in frame_detections.detections if not d.is_person]
        return (
            FrameDetections(
                frame_number=frame_detections.frame_number,
                detections=persons,
                frame_width=frame_detections.frame_width,
                frame_height=frame_detections.frame_height,
            ),
            others,
        )

    def _extract_colors(
        self, frame_detections: FrameDetections, frame: np.ndarray
    ) -> FrameDetections:
        enriched = []
        for det in frame_detections.detections:
            jersey_hist = extract_jersey_color(
                frame, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
            )
            shoe_hist = extract_shoe_color(
                frame, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
            )
            enriched.append(
                Detection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    tracking_id=det.tracking_id,
                    color_hist=jersey_hist,
                    shoe_color_hist=shoe_hist,
                )
            )
        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=enriched,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
        )

    def _filter_by_court(
        self,
        frame_detections: FrameDetections,
        frame: np.ndarray,
        court_detector: CourtDetector,
    ) -> FrameDetections:
        court_mask = court_detector.detect_court_mask(frame)
        filtered = []
        for detection in frame_detections.detections:
            is_in_court = court_detector.is_bbox_in_court(
                bbox_x=detection.bbox.x,
                bbox_y=detection.bbox.y,
                bbox_width=detection.bbox.width,
                bbox_height=detection.bbox.height,
                mask=court_mask,
                threshold=self._config.court_overlap_threshold,
            )
            if is_in_court:
                filtered.append(detection)
        return FrameDetections(
            frame_number=frame_detections.frame_number,
            detections=filtered,
            frame_width=frame_detections.frame_width,
            frame_height=frame_detections.frame_height,
        )

    async def _store_detections(
        self, video_id: int, frame_detections: FrameDetections
    ) -> dict[str, int]:
        stats = {"total": 0, "persons": 0, "balls": 0}
        for i, detection in enumerate(frame_detections.detections):
            player_detection = PlayerDetection(
                video_id=video_id,
                frame_number=frame_detections.frame_number,
                player_id=None,
                bbox_x=detection.bbox.x,
                bbox_y=detection.bbox.y,
                bbox_width=detection.bbox.width,
                bbox_height=detection.bbox.height,
                tracking_id=detection.tracking_id if detection.tracking_id is not None else i,
                confidence_score=detection.confidence,
            )
            self._db.add(player_detection)
            stats["total"] += 1
            if detection.is_person:
                stats["persons"] += 1
            else:
                stats["balls"] += 1
        return stats


class OCRBatchProcessor:
    def __init__(self, db: AsyncSession, config: BatchConfig) -> None:
        self._db = db
        self._config = config
        self._jersey_ocr: JerseyOCR | None = None
        self._legibility_config = LegibilityConfig()
        self._ocr_frame_counts: dict[int, int] = {}

    def _get_jersey_ocr(self) -> JerseyOCR:
        if self._jersey_ocr is None:
            ocr_config = OCRConfig(
                model_name=settings.ocr_model_name,
                device=self._config.device,
            )
            self._jersey_ocr = JerseyOCR(ocr_config)
        return self._jersey_ocr

    def reset_ocr_state(self) -> None:
        self._ocr_frame_counts.clear()

    def _should_run_ocr_for_track(self, tracking_id: int) -> bool:
        count = self._ocr_frame_counts.get(tracking_id, 0)
        self._ocr_frame_counts[tracking_id] = count + 1
        return count % self._config.ocr_sample_rate == 0

    async def process_batch(
        self,
        batch: ProcessingBatch,
        frames: list[np.ndarray],
        frame_numbers: list[int],
        video_id: int,
    ) -> OCRBatchResult:
        if not self._config.enable_jersey_ocr:
            batch.ocr_status = BatchStatus.SKIPPED
            await self._db.commit()
            return OCRBatchResult(batch_id=batch.id, detections_processed=0, ocr_results_created=0)

        batch.ocr_status = BatchStatus.PROCESSING
        await self._db.commit()

        work_items = await self._collect_ocr_work_items(frames, frame_numbers, video_id)
        detections_processed = work_items[0] if work_items else 0
        crops_to_process = work_items[1] if work_items else []

        if not crops_to_process:
            batch.ocr_status = BatchStatus.COMPLETED
            batch.ocr_completed_at = datetime.now(timezone.utc)
            await self._db.commit()
            return OCRBatchResult(
                batch_id=batch.id,
                detections_processed=detections_processed,
                ocr_results_created=0,
            )

        ocr = self._get_jersey_ocr()
        ocr_results = await self._run_ocr_parallel(ocr, crops_to_process)

        ocr_results_created = 0
        for item, ocr_result in zip(crops_to_process, ocr_results):
            jersey_number = JerseyNumber(
                video_id=item.video_id,
                frame_number=item.frame_number,
                tracking_id=item.tracking_id,
                raw_ocr_output=ocr_result.raw_text[:255],
                parsed_number=ocr_result.parsed_number,
                confidence=ocr_result.confidence,
                is_valid=ocr_result.is_valid,
            )
            self._db.add(jersey_number)
            ocr_results_created += 1

        await self._db.commit()

        batch.ocr_status = BatchStatus.COMPLETED
        batch.ocr_completed_at = datetime.now(timezone.utc)
        await self._db.commit()

        return OCRBatchResult(
            batch_id=batch.id,
            detections_processed=detections_processed,
            ocr_results_created=ocr_results_created,
        )

    async def _collect_ocr_work_items(
        self,
        frames: list[np.ndarray],
        frame_numbers: list[int],
        video_id: int,
    ) -> tuple[int, list[_OCRWorkItem]]:
        detections_processed = 0
        work_items: list[_OCRWorkItem] = []

        for i, frame_number in enumerate(frame_numbers):
            frame = frames[i]

            result = await self._db.execute(
                select(PlayerDetection).where(
                    PlayerDetection.video_id == video_id,
                    PlayerDetection.frame_number == frame_number,
                )
            )
            detections = result.scalars().all()

            for detection in detections:
                detections_processed += 1

                if detection.tracking_id is None:
                    continue

                if not self._should_run_ocr_for_track(detection.tracking_id):
                    continue

                bbox = BoundingBox(
                    x=detection.bbox_x,
                    y=detection.bbox_y,
                    width=detection.bbox_width,
                    height=detection.bbox_height,
                )

                legibility = check_legibility(
                    frame, bbox, detection.confidence_score, self._legibility_config
                )
                if not legibility.is_legible:
                    continue

                crop = extract_jersey_crop(frame, bbox)
                if crop.size == 0:
                    continue

                work_items.append(
                    _OCRWorkItem(
                        crop=crop,
                        video_id=video_id,
                        frame_number=frame_number,
                        tracking_id=detection.tracking_id,
                    )
                )

        return detections_processed, work_items

    async def _run_ocr_parallel(
        self,
        ocr: JerseyOCR,
        work_items: list[_OCRWorkItem],
    ) -> list[OCRResult]:
        loop = asyncio.get_running_loop()

        ocr_start_time = time.time()

        with ThreadPoolExecutor(max_workers=self._config.ocr_max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, ocr.read_jersey_number, item.crop)
                for item in work_items
            ]
            results = await asyncio.gather(*futures)

        if settings.enable_inference_timing:
            ocr_time = time.time() - ocr_start_time
            crops_per_sec = len(work_items) / ocr_time if ocr_time > 0 else 0
            logger.info(
                f"OCR batch: {len(work_items)} crops in {ocr_time:.2f}s "
                f"({crops_per_sec:.1f} crops/s, {self._config.ocr_max_workers} workers)"
            )

        return list(results)
