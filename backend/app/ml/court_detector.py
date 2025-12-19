"""Basketball court detection using computer vision.

Detects court boundaries using line detection (Hough transform) and edge detection
to create a mask for filtering out-of-court detections (like audience members).
"""

import cv2
import numpy as np
from typing import Any


class CourtDetector:
    """Detects basketball court boundaries in video frames.

    Uses edge detection and Hough line transform to identify court lines,
    then creates a polygon mask of the playing area.
    """

    def __init__(
        self,
        edge_threshold1: int = 50,
        edge_threshold2: int = 150,
        line_threshold: int = 100,
        min_line_length: int = 100,
        max_line_gap: int = 10,
    ) -> None:
        """Initialize the court detector.

        Args:
            edge_threshold1: First threshold for Canny edge detector.
            edge_threshold2: Second threshold for Canny edge detector.
            line_threshold: Accumulator threshold for Hough line detection.
            min_line_length: Minimum line length for Hough lines.
            max_line_gap: Maximum gap between line segments.
        """
        self._edge_threshold1 = edge_threshold1
        self._edge_threshold2 = edge_threshold2
        self._line_threshold = line_threshold
        self._min_line_length = min_line_length
        self._max_line_gap = max_line_gap

    def detect_court_mask(self, frame: Any) -> np.ndarray:
        """Detect court boundaries and create a binary mask.

        Args:
            frame: Input frame as numpy array (HxWxC, BGR format from OpenCV).

        Returns:
            Binary mask where court area is 255, rest is 0 (HxW).
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")

        height, width = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, self._edge_threshold1, self._edge_threshold2)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._line_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )

        if lines is None or len(lines) == 0:
            # No court detected - return full frame mask (assume all is court)
            return np.ones((height, width), dtype=np.uint8) * 255

        # Find court boundary by analyzing line positions
        # Basketball courts typically have strong horizontal and vertical lines
        boundary_points = self._extract_court_boundary(lines, width, height)

        if boundary_points is None:
            # Couldn't determine court - return full frame mask
            return np.ones((height, width), dtype=np.uint8) * 255

        # Create mask from boundary polygon
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [boundary_points], 255)

        return mask

    def _extract_court_boundary(
        self, lines: np.ndarray, width: int, height: int
    ) -> np.ndarray | None:
        """Extract court boundary polygon from detected lines.

        Args:
            lines: Detected lines from Hough transform [(x1, y1, x2, y2), ...].
            width: Frame width.
            height: Frame height.

        Returns:
            Array of boundary points [(x, y), ...] or None if can't determine.
        """
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Lines close to horizontal (0 or 180 degrees)
            if angle < 20 or angle > 160:
                horizontal_lines.append((x1, y1, x2, y2))
            # Lines close to vertical (90 degrees)
            elif 70 < angle < 110:
                vertical_lines.append((x1, y1, x2, y2))

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            # Not enough lines to determine court
            return None

        # Find extreme positions
        # Top line: minimum y among horizontal lines
        # Bottom line: maximum y among horizontal lines
        # Left line: minimum x among vertical lines
        # Right line: maximum x among vertical lines

        h_y_coords = [min(y1, y2) for x1, y1, x2, y2 in horizontal_lines]
        h_y_coords.extend([max(y1, y2) for x1, y1, x2, y2 in horizontal_lines])

        v_x_coords = [min(x1, x2) for x1, y1, x2, y2 in vertical_lines]
        v_x_coords.extend([max(x1, x2) for x1, y1, x2, y2 in vertical_lines])

        # Use percentiles to be robust against outliers
        top = int(np.percentile(h_y_coords, 10))  # 10th percentile
        bottom = int(np.percentile(h_y_coords, 90))  # 90th percentile
        left = int(np.percentile(v_x_coords, 10))
        right = int(np.percentile(v_x_coords, 90))

        # Add margin to be more inclusive (capture full court)
        margin = 20
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)

        # Create rectangular boundary
        boundary = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ], dtype=np.int32)

        return boundary

    def is_point_in_court(self, x: float, y: float, mask: np.ndarray) -> bool:
        """Check if a point is inside the court mask.

        Args:
            x: X coordinate of point.
            y: Y coordinate of point.
            mask: Court mask from detect_court_mask().

        Returns:
            True if point is inside court, False otherwise.
        """
        if mask is None:
            return True  # No mask means everything is valid

        h, w = mask.shape
        ix, iy = int(x), int(y)

        # Check bounds
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False

        return mask[iy, ix] > 0

    def is_bbox_in_court(
        self,
        bbox_x: float,
        bbox_y: float,
        bbox_width: float,
        bbox_height: float,
        mask: np.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """Check if a bounding box overlaps with the court.

        Args:
            bbox_x: Bounding box top-left x coordinate.
            bbox_y: Bounding box top-left y coordinate.
            bbox_width: Bounding box width.
            bbox_height: Bounding box height.
            mask: Court mask from detect_court_mask().
            threshold: Minimum overlap ratio (0-1) to consider in court.

        Returns:
            True if bbox overlaps court by at least threshold, False otherwise.
        """
        if mask is None:
            return True  # No mask means everything is valid

        h, w = mask.shape

        # Convert bbox to integer coordinates
        x1 = int(max(0, bbox_x))
        y1 = int(max(0, bbox_y))
        x2 = int(min(w, bbox_x + bbox_width))
        y2 = int(min(h, bbox_y + bbox_height))

        if x2 <= x1 or y2 <= y1:
            return False  # Invalid bbox

        # Extract bbox region from mask
        bbox_mask = mask[y1:y2, x1:x2]

        # Calculate overlap ratio
        court_pixels = np.sum(bbox_mask > 0)
        total_pixels = (x2 - x1) * (y2 - y1)

        if total_pixels == 0:
            return False

        overlap_ratio = court_pixels / total_pixels
        return overlap_ratio >= threshold
