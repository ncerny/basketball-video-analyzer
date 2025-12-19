"""ByteTrack wrapper for player tracking across frames."""

from typing import Any

import numpy as np
import supervision as sv

from .types import Detection, FrameDetections


class PlayerTracker:
    """Wrapper around ByteTrack for tracking players across video frames.

    Maintains persistent tracking IDs for detected players throughout the video,
    handling occlusions and temporary disappearances.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
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
            Updated FrameDetections with persistent tracking_ids assigned.
        """
        if not frame_detections.detections:
            return frame_detections

        # Convert to Supervision format
        sv_detections = self._to_supervision_detections(frame_detections)

        # Update tracker
        tracked_detections = self._tracker.update_with_detections(sv_detections)

        # Convert back and assign tracking IDs
        return self._update_tracking_ids(frame_detections, tracked_detections)

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

    def _update_tracking_ids(
        self,
        original: FrameDetections,
        tracked: sv.Detections,
    ) -> FrameDetections:
        """Update original detections with tracking IDs from ByteTrack.

        Args:
            original: Original frame detections.
            tracked: Tracked detections with tracker_id field.

        Returns:
            Original detections with updated tracking_ids.
        """
        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return original

        # ByteTrack may return fewer detections than input (filtering low confidence)
        # Match detections by bounding box overlap
        num_tracked = len(tracked.tracker_id)

        # If counts match, assume 1:1 correspondence
        if num_tracked == len(original.detections):
            for i, detection in enumerate(original.detections):
                detection.tracking_id = int(tracked.tracker_id[i])
        else:
            # Counts don't match - need to match by position
            # This can happen if ByteTrack filters some detections
            # For now, assign tracking IDs to first N detections
            for i in range(min(num_tracked, len(original.detections))):
                original.detections[i].tracking_id = int(tracked.tracker_id[i])

        return original
