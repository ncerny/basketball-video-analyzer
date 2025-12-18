"""Tests for ByteTrack player tracker module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.byte_tracker import PlayerTracker
from app.ml.types import BoundingBox, Detection, FrameDetections


class TestPlayerTracker:
    """Tests for PlayerTracker class."""

    def test_init_default_values(self) -> None:
        """Test tracker initialization with default values."""
        tracker = PlayerTracker()
        assert tracker._tracker is not None

    def test_init_custom_values(self) -> None:
        """Test tracker initialization with custom values."""
        tracker = PlayerTracker(
            track_activation_threshold=0.7,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=60,
        )
        assert tracker._tracker is not None

    @patch("app.ml.byte_tracker.sv.ByteTrack")
    def test_init_passes_params_to_bytetrack(self, mock_bytetrack: MagicMock) -> None:
        """Test that initialization parameters are passed to ByteTrack."""
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance

        tracker = PlayerTracker(
            track_activation_threshold=0.6,
            lost_track_buffer=45,
            minimum_matching_threshold=0.75,
            frame_rate=30,
        )

        mock_bytetrack.assert_called_once_with(
            track_activation_threshold=0.6,
            lost_track_buffer=45,
            minimum_matching_threshold=0.75,
            frame_rate=30,
        )

    def test_update_empty_detections(self) -> None:
        """Test update with empty detections."""
        tracker = PlayerTracker()
        frame_detections = FrameDetections(frame_number=0, detections=[])

        result = tracker.update(frame_detections)

        assert result.frame_number == 0
        assert len(result.detections) == 0

    @patch("app.ml.byte_tracker.sv.ByteTrack")
    def test_update_assigns_tracking_ids(self, mock_bytetrack: MagicMock) -> None:
        """Test that update assigns tracking IDs from ByteTrack."""
        # Create mock tracker
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance

        # Mock tracked detections with tracking IDs
        mock_tracked = MagicMock()
        mock_tracked.tracker_id = np.array([1, 2, 3])
        mock_tracker_instance.update_with_detections.return_value = mock_tracked

        # Create test detections
        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        detections = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.85, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.8, class_id=0, class_name="person"),
        ]
        frame_detections = FrameDetections(
            frame_number=5, detections=detections, frame_width=1920, frame_height=1080
        )

        # Run update
        tracker = PlayerTracker()
        result = tracker.update(frame_detections)

        # Verify tracking IDs were assigned
        assert result.detections[0].tracking_id == 1
        assert result.detections[1].tracking_id == 2
        assert result.detections[2].tracking_id == 3

    @patch("app.ml.byte_tracker.sv.ByteTrack")
    def test_update_handles_no_tracker_id(self, mock_bytetrack: MagicMock) -> None:
        """Test update when ByteTrack returns None for tracker_id."""
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance

        # Mock tracked detections with no tracking IDs
        mock_tracked = MagicMock()
        mock_tracked.tracker_id = None
        mock_tracker_instance.update_with_detections.return_value = mock_tracked

        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        detections = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
        ]
        frame_detections = FrameDetections(frame_number=1, detections=detections)

        tracker = PlayerTracker()
        result = tracker.update(frame_detections)

        # Should return original without modification
        assert result.detections[0].tracking_id is None

    @patch("app.ml.byte_tracker.sv.ByteTrack")
    def test_reset(self, mock_bytetrack: MagicMock) -> None:
        """Test tracker reset."""
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance

        tracker = PlayerTracker()
        tracker.reset()

        mock_tracker_instance.reset.assert_called_once()

    def test_to_supervision_detections_conversion(self) -> None:
        """Test conversion from FrameDetections to Supervision Detections."""
        tracker = PlayerTracker()

        # Create test detections with varied bounding boxes
        detections = [
            Detection(
                bbox=BoundingBox(x=10, y=20, width=100, height=200),
                confidence=0.9,
                class_id=0,
                class_name="person",
            ),
            Detection(
                bbox=BoundingBox(x=200, y=100, width=80, height=160),
                confidence=0.85,
                class_id=0,
                class_name="person",
            ),
        ]
        frame_detections = FrameDetections(
            frame_number=1, detections=detections, frame_width=1920, frame_height=1080
        )

        sv_detections = tracker._to_supervision_detections(frame_detections)

        # Verify conversion
        assert len(sv_detections) == 2
        assert sv_detections.confidence[0] == 0.9
        assert sv_detections.confidence[1] == 0.85
        assert sv_detections.class_id[0] == 0
        assert sv_detections.class_id[1] == 0

        # Verify xyxy conversion
        np.testing.assert_array_equal(sv_detections.xyxy[0], [10, 20, 110, 220])
        np.testing.assert_array_equal(sv_detections.xyxy[1], [200, 100, 280, 260])

    def test_to_supervision_detections_empty(self) -> None:
        """Test conversion with empty detections."""
        tracker = PlayerTracker()
        frame_detections = FrameDetections(frame_number=0, detections=[])

        sv_detections = tracker._to_supervision_detections(frame_detections)

        assert len(sv_detections) == 0

    @patch("app.ml.byte_tracker.sv.ByteTrack")
    def test_tracking_persistence_across_frames(self, mock_bytetrack: MagicMock) -> None:
        """Test that tracking IDs persist across multiple frames."""
        mock_tracker_instance = MagicMock()
        mock_bytetrack.return_value = mock_tracker_instance

        # Simulate same player tracked across frames with same ID
        mock_tracked1 = MagicMock()
        mock_tracked1.tracker_id = np.array([1, 2])

        mock_tracked2 = MagicMock()
        mock_tracked2.tracker_id = np.array([1, 2])  # Same IDs

        mock_tracker_instance.update_with_detections.side_effect = [mock_tracked1, mock_tracked2]

        tracker = PlayerTracker()
        bbox = BoundingBox(x=10, y=20, width=100, height=200)

        # Frame 1
        detections1 = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.85, class_id=0, class_name="person"),
        ]
        frame1 = FrameDetections(frame_number=1, detections=detections1)
        result1 = tracker.update(frame1)

        # Frame 2
        detections2 = [
            Detection(bbox=bbox, confidence=0.88, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.83, class_id=0, class_name="person"),
        ]
        frame2 = FrameDetections(frame_number=2, detections=detections2)
        result2 = tracker.update(frame2)

        # Verify same tracking IDs
        assert result1.detections[0].tracking_id == 1
        assert result1.detections[1].tracking_id == 2
        assert result2.detections[0].tracking_id == 1
        assert result2.detections[1].tracking_id == 2
