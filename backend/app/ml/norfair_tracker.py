"""Norfair tracker with jersey color Re-ID for handling player crossings."""

import numpy as np
from norfair import Detection as NorfairDetection
from norfair import Tracker

from .color_extractor import color_similarity
from .types import BoundingBox, Detection, FrameDetections


CLASS_NAMES: dict[int, str] = {
    0: "person",
    32: "sports_ball",
}


def _reid_distance(obj1, obj2) -> float:
    """Compare jersey colors between two tracked objects. Returns 0-1 (lower = more similar)."""
    hist1 = getattr(obj1.last_detection, "data", {}).get("color_hist")
    hist2 = getattr(obj2.last_detection, "data", {}).get("color_hist")

    if hist1 is None or hist2 is None:
        return 1.0

    similarity = color_similarity(hist1, hist2)
    return 1.0 - similarity


class NorfairTracker:
    def __init__(
        self,
        distance_threshold: float = 250.0,
        hit_counter_max: int = 90,
        initialization_delay: int = 1,
        pointwise_hit_counter_max: int = 4,
        reid_distance_threshold: float = 0.5,
        reid_hit_counter_max: int = 150,
    ) -> None:
        self._distance_threshold = distance_threshold
        self._hit_counter_max = hit_counter_max
        self._initialization_delay = initialization_delay
        self._pointwise_hit_counter_max = pointwise_hit_counter_max
        self._reid_distance_threshold = reid_distance_threshold
        self._reid_hit_counter_max = reid_hit_counter_max

        self._tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=pointwise_hit_counter_max,
            reid_distance_function=_reid_distance,
            reid_distance_threshold=reid_distance_threshold,
            reid_hit_counter_max=reid_hit_counter_max,
        )

    def update(self, frame_detections: FrameDetections) -> FrameDetections:
        norfair_detections = self._to_norfair_detections(frame_detections)
        tracked_objects = self._tracker.update(detections=norfair_detections)
        return self._from_norfair_objects(frame_detections, tracked_objects)

    def reset(self) -> None:
        self._tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=self._distance_threshold,
            hit_counter_max=self._hit_counter_max,
            initialization_delay=self._initialization_delay,
            pointwise_hit_counter_max=self._pointwise_hit_counter_max,
            reid_distance_function=_reid_distance,
            reid_distance_threshold=self._reid_distance_threshold,
            reid_hit_counter_max=self._reid_hit_counter_max,
        )

    def _to_norfair_detections(self, frame_detections: FrameDetections) -> list[NorfairDetection]:
        norfair_dets = []

        for det in frame_detections.detections:
            center_x = det.bbox.x + det.bbox.width / 2
            center_y = det.bbox.y + det.bbox.height / 2
            points = np.array([[center_x, center_y]])
            scores = np.array([det.confidence])

            norfair_det = NorfairDetection(
                points=points,
                scores=scores,
                label=det.class_id,
                data={
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "color_hist": det.color_hist,
                },
            )
            norfair_dets.append(norfair_det)

        return norfair_dets

    def _from_norfair_objects(
        self,
        original: FrameDetections,
        tracked_objects: list,
    ) -> FrameDetections:
        detections: list[Detection] = []

        for obj in tracked_objects:
            if obj.last_detection is None or obj.last_detection.data is None:
                continue

            data = obj.last_detection.data
            bbox = data.get("bbox")
            if bbox is None:
                continue

            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=data.get("confidence", 0.0),
                    class_id=data.get("class_id", 0),
                    class_name=data.get("class_name", "unknown"),
                    tracking_id=obj.id,
                    color_hist=data.get("color_hist"),
                )
            )

        return FrameDetections(
            frame_number=original.frame_number,
            detections=detections,
            frame_width=original.frame_width,
            frame_height=original.frame_height,
        )
