"""Color extraction for player Re-ID (jersey and shoe colors)."""

import cv2
import numpy as np


def extract_jersey_color(
    frame: np.ndarray, bbox_x: float, bbox_y: float, bbox_width: float, bbox_height: float
) -> np.ndarray:
    """Extract HSV color histogram from upper portion of player bbox (jersey area).

    Returns a normalized 48-bin histogram (16 hue + 16 saturation + 16 value).
    """
    x1 = max(0, int(bbox_x))
    y1 = max(0, int(bbox_y))
    x2 = min(frame.shape[1], int(bbox_x + bbox_width))
    y2 = min(frame.shape[0], int(bbox_y + bbox_height))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)

    jersey_y2 = y1 + int((y2 - y1) * 0.5)
    crop = frame[y1:jersey_y2, x1:x2]

    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    return hist.astype(np.float32)


def extract_shoe_color(
    frame: np.ndarray, bbox_x: float, bbox_y: float, bbox_width: float, bbox_height: float
) -> np.ndarray:
    """Extract HSV color histogram from lower portion of player bbox (shoe area).

    Extracts from bottom 20% of the bounding box where shoes are typically visible.
    Returns a normalized 48-bin histogram (16 hue + 16 saturation + 16 value).
    """
    x1 = max(0, int(bbox_x))
    y1 = max(0, int(bbox_y))
    x2 = min(frame.shape[1], int(bbox_x + bbox_width))
    y2 = min(frame.shape[0], int(bbox_y + bbox_height))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)

    # Extract bottom 20% of bbox (shoe area)
    shoe_y1 = y1 + int((y2 - y1) * 0.8)
    crop = frame[shoe_y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    return hist.astype(np.float32)


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
