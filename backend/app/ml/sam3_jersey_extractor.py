"""Jersey crop extraction using SAM3 masks.

SAM3 provides precise segmentation masks that can be used to
extract jersey regions while removing background pixels.
"""

import numpy as np

from .types import BoundingBox


def extract_jersey_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    mask: np.ndarray | None = None,
    jersey_height_ratio: float = 0.4,
) -> np.ndarray:
    """Extract jersey region from frame using optional mask.

    Args:
        frame: BGR frame image.
        bbox: Player bounding box.
        mask: Optional segmentation mask (same size as frame).
        jersey_height_ratio: Fraction of bbox height for jersey (default: 40%).

    Returns:
        Cropped jersey region with background zeroed if mask provided.
    """
    x1, y1, x2, y2 = bbox.to_xyxy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Jersey is upper portion of player
    jersey_y2 = y1 + int((y2 - y1) * jersey_height_ratio)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    jersey_y2 = min(frame.shape[0], jersey_y2)

    # Validate dimensions after clamping
    if x2 <= x1 or jersey_y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    # Extract crop
    crop = frame[y1:jersey_y2, x1:x2].copy()

    # Apply mask if provided
    if mask is not None:
        crop_mask = mask[y1:jersey_y2, x1:x2].astype(bool)
        # Zero out background pixels
        crop[~crop_mask] = 0

    return crop
