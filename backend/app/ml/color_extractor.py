"""Color extraction for player Re-ID (jersey and shoe colors)."""

import cv2
import numpy as np


def extract_jersey_color(
    frame: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract HSV color histogram from player bbox, optionally masked.

    Args:
        frame: BGR image.
        bbox_x, bbox_y, bbox_width, bbox_height: Bounding box coordinates.
        mask: Optional boolean mask (same size as frame). If provided, only
              pixels where mask is True are included in histogram.

    Returns:
        Normalized 48-bin histogram (16 hue + 16 saturation + 16 value).
        Using full body captures jersey, shorts, skin tone, and shoes for better
        individual player discrimination while maintaining team clustering.
    """
    x1 = max(0, int(bbox_x))
    y1 = max(0, int(bbox_y))
    x2 = min(frame.shape[1], int(bbox_x + bbox_width))
    y2 = min(frame.shape[0], int(bbox_y + bbox_height))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)

    # Crop frame to bbox
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    # Crop mask to bbox if provided
    mask_crop = None
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2].astype(np.uint8)
        # Check if mask has any valid pixels
        if mask_crop.sum() == 0:
            return np.zeros(48, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Use mask_crop for histogram calculation (None means use all pixels)
    h_hist = cv2.calcHist([hsv], [0], mask_crop, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask_crop, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask_crop, [16], [0, 256])

    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    return hist.astype(np.float32)


def extract_shoe_color(
    frame: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract HSV color histogram from full player bbox, optionally masked.

    NOTE: Now uses full body (same as extract_jersey_color) for better
    player discrimination. Kept as separate function for API compatibility.
    """
    return extract_jersey_color(frame, bbox_x, bbox_y, bbox_width, bbox_height, mask)


def extract_combined_colors(
    frame: np.ndarray, bbox_x: float, bbox_y: float, bbox_width: float, bbox_height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both jersey and shoe color histograms for a player detection.

    Returns:
        Tuple of (jersey_hist, shoe_hist), each a 48-bin normalized histogram.
    """
    jersey_hist = extract_jersey_color(frame, bbox_x, bbox_y, bbox_width, bbox_height)
    shoe_hist = extract_shoe_color(frame, bbox_x, bbox_y, bbox_width, bbox_height)
    return jersey_hist, shoe_hist


def color_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute cosine similarity between two color histograms. Returns 0-1."""
    if hist1 is None or hist2 is None:
        return 0.0

    dot = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))
