"""ML module for basketball video analysis.

Provides player detection and tracking using SAM3 (Segment Anything Model 3).
"""

from .base import BaseDetector
from .types import BoundingBox, Detection, DetectionClass, FrameDetections

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionClass",
    "FrameDetections",
]
