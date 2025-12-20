from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DetectionClass(Enum):
    PERSON = 0
    SPORTS_BALL = 32


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def x_center(self) -> float:
        return self.x + self.width / 2

    @property
    def y_center(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    tracking_id: int | None = None
    color_hist: np.ndarray | None = None

    @property
    def is_person(self) -> bool:
        return self.class_id == DetectionClass.PERSON.value


@dataclass
class FrameDetections:
    frame_number: int
    detections: list[Detection] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.is_person)
