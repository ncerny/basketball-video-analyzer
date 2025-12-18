"""ML module for basketball video analysis.

Provides player detection using YOLOv8 models.
"""

from .base import BaseDetector
from .types import BoundingBox, Detection, DetectionClass, FrameDetections
from .yolo_detector import YOLODetector

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionClass",
    "FrameDetections",
    "YOLODetector",
]
