import logging
from typing import Any

import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

from .base import BaseDetector
from .types import BoundingBox, Detection, DetectionClass, FrameDetections

logger = logging.getLogger(__name__)


class RFDETRDetector(BaseDetector):
    RFDETR_PERSON_CLASS_ID = 1
    RFDETR_SPORTS_BALL_CLASS_ID = 37
    VALID_RFDETR_CLASS_IDS = {RFDETR_PERSON_CLASS_ID, RFDETR_SPORTS_BALL_CLASS_ID}

    OUTPUT_PERSON_CLASS_ID = DetectionClass.PERSON.value
    OUTPUT_SPORTS_BALL_CLASS_ID = DetectionClass.SPORTS_BALL.value

    RFDETR_TO_OUTPUT_CLASS = {
        RFDETR_PERSON_CLASS_ID: OUTPUT_PERSON_CLASS_ID,
        RFDETR_SPORTS_BALL_CLASS_ID: OUTPUT_SPORTS_BALL_CLASS_ID,
    }

    CLASS_NAMES = {
        OUTPUT_PERSON_CLASS_ID: "person",
        OUTPUT_SPORTS_BALL_CLASS_ID: "sports_ball",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: RFDETRBase | None = None

    def _load_model(self) -> None:
        if self._model is None:
            logger.info(f"Loading RF-DETR model on device: {self._device}")
            self._model = RFDETRBase()
            if self._device == "cuda":
                try:
                    self._model.optimize_for_inference()
                    logger.info("RF-DETR model loaded and optimized for CUDA")
                except Exception as e:
                    logger.warning(f"Could not optimize RF-DETR for inference: {e}")
            else:
                logger.info("RF-DETR model loaded (optimization skipped for non-CUDA device)")

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> dict[str, str | int | float]:
        return {
            "model_type": "RF-DETR Base",
            "device": self._device,
            "confidence_threshold": self._confidence_threshold,
        }

    def detect(self, frame: Any, frame_number: int = 0) -> FrameDetections:
        self._load_model()
        assert self._model is not None

        image = Image.fromarray(frame[:, :, ::-1])
        sv_detections = self._model.predict(image, threshold=self._confidence_threshold)

        detections = self._convert_sv_detections(sv_detections)

        if isinstance(frame, np.ndarray):
            height, width = frame.shape[:2]
        else:
            height, width = 0, 0

        return FrameDetections(
            frame_number=frame_number,
            detections=detections,
            frame_width=width,
            frame_height=height,
        )

    def detect_batch(self, frames: list[Any], start_frame_number: int = 0) -> list[FrameDetections]:
        self._load_model()
        assert self._model is not None

        if not frames:
            return []

        if start_frame_number == 0:
            logger.info(f"Starting RF-DETR batch inference on device: {self._device}")

        images = [Image.fromarray(f[:, :, ::-1]) for f in frames]
        detections_list = self._model.predict(images, threshold=self._confidence_threshold)

        frame_detections = []
        for i, sv_detections in enumerate(detections_list):
            detections = self._convert_sv_detections(sv_detections)

            frame = frames[i]
            if isinstance(frame, np.ndarray):
                height, width = frame.shape[:2]
            else:
                height, width = 0, 0

            frame_detections.append(
                FrameDetections(
                    frame_number=start_frame_number + i,
                    detections=detections,
                    frame_width=width,
                    frame_height=height,
                )
            )

        return frame_detections

    def _convert_sv_detections(self, sv_detections) -> list[Detection]:
        detections = []

        if sv_detections is None or len(sv_detections) == 0:
            return detections

        for i in range(len(sv_detections)):
            rfdetr_class_id = int(sv_detections.class_id[i])

            if rfdetr_class_id not in self.VALID_RFDETR_CLASS_IDS:
                continue

            output_class_id = self.RFDETR_TO_OUTPUT_CLASS[rfdetr_class_id]

            x1, y1, x2, y2 = sv_detections.xyxy[i]
            confidence = float(sv_detections.confidence[i])

            detection = Detection(
                bbox=BoundingBox.from_xyxy(float(x1), float(y1), float(x2), float(y2)),
                confidence=confidence,
                class_id=output_class_id,
                class_name=self.CLASS_NAMES.get(output_class_id, f"class_{output_class_id}"),
                tracking_id=None,
            )
            detections.append(detection)

        return detections

    def set_confidence_threshold(self, threshold: float) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self._confidence_threshold = threshold

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def device(self) -> str:
        return self._device
