"""ByteTrack wrapper for player tracking across frames."""

import numpy as np
import supervision as sv

from .types import BoundingBox, Detection, FrameDetections


# Reverse mapping from YOLO class IDs to class names
CLASS_NAMES: dict[int, str] = {
    0: "person",
    32: "sports_ball",
}


class PlayerTracker:
    """Wrapper around ByteTrack for tracking players across video frames.

    Maintains persistent tracking IDs for detected players throughout the video,
    handling occlusions and temporary disappearances.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.35,
        frame_rate: int = 30,
    ) -> None:
        """Initialize the player tracker.

        Args:
            track_activation_threshold: Min confidence to start new track (0-1).
            lost_track_buffer: Frames to keep lost tracks alive (handle occlusions).
            minimum_matching_threshold: IOU threshold for matching (0-1).
            frame_rate: Video frame rate for motion prediction.
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, frame_detections: FrameDetections) -> FrameDetections:
        """Update tracker with new frame detections.

        Args:
            frame_detections: Detections from current frame (from YOLO).

        Returns:
            FrameDetections with tracked detections and persistent tracking_ids.
            Note: The returned detections come from ByteTrack's output, which may
            differ from the input (filtered, reordered, or empty).
        """
        # Convert to Supervision format
        sv_detections = self._to_supervision_detections(frame_detections)

        # Always update tracker, even with empty detections, to properly age lost tracks
        tracked_detections = self._tracker.update_with_detections(sv_detections)

        # Convert tracked results back to our format (this is the source of truth)
        return self._from_supervision_detections(frame_detections, tracked_detections)

    def reset(self) -> None:
        """Reset tracker state (for new video)."""
        self._tracker.reset()

    def _to_supervision_detections(self, frame_detections: FrameDetections) -> sv.Detections:
        """Convert FrameDetections to Supervision Detections format.

        Args:
            frame_detections: Our internal format.

        Returns:
            Supervision Detections object.
        """
        if not frame_detections.detections:
            return sv.Detections.empty()

        # Extract bounding boxes in xyxy format
        xyxy = np.array([det.bbox.to_xyxy() for det in frame_detections.detections])

        # Extract confidence scores
        confidence = np.array([det.confidence for det in frame_detections.detections])

        # Extract class IDs
        class_id = np.array([det.class_id for det in frame_detections.detections])

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def _from_supervision_detections(
        self,
        original: FrameDetections,
        tracked: sv.Detections,
    ) -> FrameDetections:
        """Convert tracked sv.Detections back to our FrameDetections format.

        This creates new Detection objects from ByteTrack's output, which is the
        source of truth for tracked detections. ByteTrack may filter, reorder,
        or modify detections, so we cannot assume index alignment with the input.

        Args:
            original: Original frame detections (used for frame metadata only).
            tracked: Tracked detections from ByteTrack with tracker_id field.

        Returns:
            New FrameDetections with detections converted from ByteTrack output.
        """
        # Handle empty tracked results
        if len(tracked) == 0:
            return FrameDetections(
                frame_number=original.frame_number,
                detections=[],
                frame_width=original.frame_width,
                frame_height=original.frame_height,
            )

        detections: list[Detection] = []
        for i in range(len(tracked)):
            # Extract bbox in xyxy format and convert to our BoundingBox
            x1, y1, x2, y2 = tracked.xyxy[i]
            bbox = BoundingBox.from_xyxy(float(x1), float(y1), float(x2), float(y2))

            # Get class info
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            class_name = CLASS_NAMES.get(class_id, "unknown")

            # Get confidence
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Get tracking ID
            tracking_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    tracking_id=tracking_id,
                )
            )

        return FrameDetections(
            frame_number=original.frame_number,
            detections=detections,
            frame_width=original.frame_width,
            frame_height=original.frame_height,
        )
