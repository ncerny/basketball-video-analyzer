"""Tests for SAM3 video tracker."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.ml.types import BoundingBox, Detection, FrameDetections


class TestSAM3TrackerConfig:
    """Tests for SAM3TrackerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from app.ml.sam3_tracker import SAM3TrackerConfig

        config = SAM3TrackerConfig()
        assert config.prompt == "basketball player"
        assert config.confidence_threshold == 0.25
        assert config.device == "auto"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        from app.ml.sam3_tracker import SAM3TrackerConfig

        config = SAM3TrackerConfig(
            prompt="player in white",
            confidence_threshold=0.5,
            device="cuda",
        )
        assert config.prompt == "player in white"
        assert config.confidence_threshold == 0.5
        assert config.device == "cuda"


class TestSAM3VideoTracker:
    """Tests for SAM3VideoTracker class."""

    def test_init_creates_tracker(self) -> None:
        """Test that initialization creates tracker instance."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig()
        tracker = SAM3VideoTracker(config)

        assert tracker._config == config
        assert tracker._model is None  # Lazy loaded
        assert tracker._processor is None  # Lazy loaded

    def test_select_device_prefers_cuda(self) -> None:
        """Test device selection prefers CUDA when available."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "cuda"

    def test_select_device_falls_back_to_mps(self) -> None:
        """Test device selection falls back to MPS when CUDA unavailable."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.zeros.return_value = MagicMock()  # MPS test succeeds

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "mps"

    def test_select_device_falls_back_to_cpu(self) -> None:
        """Test device selection falls back to CPU as last resort."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "cpu"

    def test_convert_to_frame_detections(self) -> None:
        """Test conversion from SAM3 output to FrameDetections."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig()
        tracker = SAM3VideoTracker(config)

        # Mock SAM3 output format
        sam3_output = {
            "frame_index": 5,
            "object_ids": [1, 2, 3],
            "boxes": np.array([
                [10, 20, 100, 200],  # x1, y1, x2, y2
                [200, 100, 300, 250],
                [400, 150, 500, 350],
            ]),
            "scores": np.array([0.95, 0.88, 0.75]),
            "masks": [
                np.ones((200, 100), dtype=bool),
                np.ones((150, 100), dtype=bool),
                np.ones((200, 100), dtype=bool),
            ],
        }

        result = tracker._convert_to_frame_detections(
            sam3_output,
            frame_number=15,  # Original frame number (before sampling)
            frame_width=1920,
            frame_height=1080,
        )

        assert isinstance(result, FrameDetections)
        assert result.frame_number == 15
        assert len(result.detections) == 3
        assert result.detections[0].tracking_id == 1
        assert result.detections[1].tracking_id == 2
        assert result.detections[2].tracking_id == 3
        assert result.detections[0].confidence == 0.95

    def test_convert_filters_low_confidence(self) -> None:
        """Test that conversion filters out low confidence detections."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig(confidence_threshold=0.5)
        tracker = SAM3VideoTracker(config)

        sam3_output = {
            "frame_index": 0,
            "object_ids": [1, 2],
            "boxes": np.array([[10, 20, 100, 200], [200, 100, 300, 250]]),
            "scores": np.array([0.8, 0.3]),  # Second one below threshold
            "masks": [np.ones((100, 100), dtype=bool)] * 2,
        }

        result = tracker._convert_to_frame_detections(
            sam3_output, frame_number=0, frame_width=1920, frame_height=1080
        )

        assert len(result.detections) == 1
        assert result.detections[0].tracking_id == 1
