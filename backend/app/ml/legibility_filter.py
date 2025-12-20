"""Heuristic legibility filter for jersey number OCR.

Filters out crops unlikely to produce readable jersey numbers before sending to OCR.
Reduces wasted OCR calls by ~50%+ on blurry/small/low-confidence detections.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .types import BoundingBox


@dataclass
class LegibilityResult:
    is_legible: bool
    score: float
    reasons: list[str]


@dataclass
class LegibilityConfig:
    min_bbox_area: int = 2000
    min_confidence: float = 0.7
    min_blur_variance: float = 50.0
    min_aspect_ratio: float = 0.8
    max_aspect_ratio: float = 4.0


def compute_blur_score(crop: np.ndarray) -> float:
    """Laplacian variance blur detection. Higher value = sharper image."""
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def extract_jersey_crop(
    frame: np.ndarray, bbox: BoundingBox, jersey_fraction: float = 0.5
) -> np.ndarray:
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(frame.shape[1], int(bbox.x + bbox.width))
    y2 = min(frame.shape[0], int(bbox.y + bbox.height))

    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=np.uint8)

    jersey_y2 = y1 + int((y2 - y1) * jersey_fraction)
    return frame[y1:jersey_y2, x1:x2]


def check_legibility(
    frame: np.ndarray,
    bbox: BoundingBox,
    confidence: float,
    config: LegibilityConfig | None = None,
) -> LegibilityResult:
    if config is None:
        config = LegibilityConfig()

    reasons: list[str] = []
    scores: list[float] = []

    area = bbox.area
    if area < config.min_bbox_area:
        reasons.append(f"bbox_too_small: {area:.0f} < {config.min_bbox_area}")
        scores.append(0.0)
    else:
        area_score = min(1.0, area / (config.min_bbox_area * 4))
        scores.append(area_score)

    if confidence < config.min_confidence:
        reasons.append(f"low_confidence: {confidence:.2f} < {config.min_confidence}")
        scores.append(0.0)
    else:
        scores.append(confidence)

    aspect_ratio = bbox.height / bbox.width if bbox.width > 0 else 0
    if aspect_ratio < config.min_aspect_ratio:
        reasons.append(f"aspect_ratio_too_low: {aspect_ratio:.2f} < {config.min_aspect_ratio}")
        scores.append(0.0)
    elif aspect_ratio > config.max_aspect_ratio:
        reasons.append(f"aspect_ratio_too_high: {aspect_ratio:.2f} > {config.max_aspect_ratio}")
        scores.append(0.0)
    else:
        if aspect_ratio < 1.5:
            ar_score = (aspect_ratio - config.min_aspect_ratio) / (1.5 - config.min_aspect_ratio)
        elif aspect_ratio > 2.5:
            ar_score = (config.max_aspect_ratio - aspect_ratio) / (config.max_aspect_ratio - 2.5)
        else:
            ar_score = 1.0
        scores.append(max(0.0, ar_score))

    crop = extract_jersey_crop(frame, bbox)
    if crop.size == 0:
        reasons.append("empty_crop")
        scores.append(0.0)
    else:
        blur_score = compute_blur_score(crop)
        if blur_score < config.min_blur_variance:
            reasons.append(f"too_blurry: {blur_score:.1f} < {config.min_blur_variance}")
            scores.append(0.0)
        else:
            blur_normalized = min(1.0, blur_score / (config.min_blur_variance * 4))
            scores.append(blur_normalized)

    final_score = sum(scores) / len(scores) if scores else 0.0
    is_legible = len(reasons) == 0

    return LegibilityResult(is_legible=is_legible, score=final_score, reasons=reasons)


def filter_legible_detections(
    frame: np.ndarray,
    bboxes: list[BoundingBox],
    confidences: list[float],
    config: LegibilityConfig | None = None,
) -> list[tuple[int, LegibilityResult]]:
    results: list[tuple[int, LegibilityResult]] = []

    for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        result = check_legibility(frame, bbox, conf, config)
        if result.is_legible:
            results.append((i, result))

    return results
