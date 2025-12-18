"""Tests for YOLO detector module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.types import BoundingBox, Detection, DetectionClass, FrameDetections
from app.ml.yolo_detector import YOLODetector


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_create_bounding_box(self) -> None:
        """Test basic bounding box creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_bounding_box_center(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x_center == 60  # 10 + 100/2
        assert bbox.y_center == 45  # 20 + 50/2

    def test_bounding_box_area(self) -> None:
        """Test area calculation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        assert bbox.area == 5000

    def test_to_xyxy(self) -> None:
        """Test conversion to xyxy format."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        x1, y1, x2, y2 = bbox.to_xyxy()
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110  # 10 + 100
        assert y2 == 70  # 20 + 50

    def test_from_xyxy(self) -> None:
        """Test creation from xyxy format."""
        bbox = BoundingBox.from_xyxy(10, 20, 110, 70)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50


class TestDetection:
    """Tests for Detection dataclass."""

    def test_create_detection(self) -> None:
        """Test basic detection creation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert detection.bbox == bbox
        assert detection.confidence == 0.95
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.tracking_id is None

    def test_detection_with_tracking_id(self) -> None:
        """Test detection with tracking ID."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        detection = Detection(
            bbox=bbox,
            confidence=0.85,
            class_id=0,
            class_name="person",
            tracking_id=42,
        )
        assert detection.tracking_id == 42

    def test_is_person(self) -> None:
        """Test is_person property."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)

        person = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person")
        assert person.is_person is True

        ball = Detection(bbox=bbox, confidence=0.9, class_id=32, class_name="sports_ball")
        assert ball.is_person is False


class TestFrameDetections:
    """Tests for FrameDetections dataclass."""

    def test_create_frame_detections(self) -> None:
        """Test basic frame detections creation."""
        fd = FrameDetections(frame_number=0)
        assert fd.frame_number == 0
        assert fd.detections == []
        assert fd.frame_width == 0
        assert fd.frame_height == 0

    def test_frame_detections_with_data(self) -> None:
        """Test frame detections with actual data."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        detection = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person")

        fd = FrameDetections(
            frame_number=42,
            detections=[detection],
            frame_width=1920,
            frame_height=1080,
        )
        assert fd.frame_number == 42
        assert len(fd.detections) == 1
        assert fd.frame_width == 1920
        assert fd.frame_height == 1080

    def test_person_count(self) -> None:
        """Test person count calculation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        persons = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.85, class_id=0, class_name="person"),
        ]
        ball = Detection(bbox=bbox, confidence=0.8, class_id=32, class_name="sports_ball")

        fd = FrameDetections(
            frame_number=0,
            detections=persons + [ball],
        )
        assert fd.person_count == 2


class TestDetectionClass:
    """Tests for DetectionClass enum."""

    def test_person_class(self) -> None:
        """Test person class ID."""
        assert DetectionClass.PERSON.value == 0

    def test_sports_ball_class(self) -> None:
        """Test sports ball class ID."""
        assert DetectionClass.SPORTS_BALL.value == 32


class TestYOLODetector:
    """Tests for YOLODetector class."""

    def test_init_default_values(self) -> None:
        """Test detector initialization with default values."""
        detector = YOLODetector()
        assert detector._model_path == "yolov8n.pt"
        assert detector._confidence_threshold == 0.5
        assert detector._device == "cpu"
        assert detector._model is None

    def test_init_custom_values(self) -> None:
        """Test detector initialization with custom values."""
        detector = YOLODetector(
            model_path="custom_model.pt",
            confidence_threshold=0.7,
            device="cuda",
        )
        assert detector._model_path == "custom_model.pt"
        assert detector._confidence_threshold == 0.7
        assert detector._device == "cuda"

    def test_is_loaded_before_load(self) -> None:
        """Test is_loaded returns False before loading."""
        detector = YOLODetector()
        assert detector.is_loaded() is False

    @patch("app.ml.yolo_detector.YOLO")
    def test_is_loaded_after_load(self, mock_yolo: MagicMock) -> None:
        """Test is_loaded returns True after loading."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = YOLODetector()
        detector._load_model()

        assert detector.is_loaded() is True

    def test_set_confidence_threshold_valid(self) -> None:
        """Test setting valid confidence threshold."""
        detector = YOLODetector()
        detector.set_confidence_threshold(0.8)
        assert detector.confidence_threshold == 0.8

    def test_set_confidence_threshold_boundaries(self) -> None:
        """Test confidence threshold at boundaries."""
        detector = YOLODetector()

        detector.set_confidence_threshold(0.0)
        assert detector.confidence_threshold == 0.0

        detector.set_confidence_threshold(1.0)
        assert detector.confidence_threshold == 1.0

    def test_set_confidence_threshold_invalid(self) -> None:
        """Test setting invalid confidence threshold."""
        detector = YOLODetector()

        with pytest.raises(ValueError, match="between 0 and 1"):
            detector.set_confidence_threshold(-0.1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            detector.set_confidence_threshold(1.1)

    def test_properties(self) -> None:
        """Test property accessors."""
        detector = YOLODetector(confidence_threshold=0.6, device="mps")
        assert detector.confidence_threshold == 0.6
        assert detector.device == "mps"

    @patch("app.ml.yolo_detector.YOLO")
    def test_get_model_info(self, mock_yolo: MagicMock) -> None:
        """Test getting model information."""
        mock_model = MagicMock()
        mock_model.type = "detect"
        mock_model.task = "detect"
        mock_yolo.return_value = mock_model

        detector = YOLODetector(model_path="test.pt", confidence_threshold=0.6)
        info = detector.get_model_info()

        assert info["model_path"] == "test.pt"
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.6

    @patch("app.ml.yolo_detector.YOLO")
    def test_detect_single_frame(self, mock_yolo: MagicMock) -> None:
        """Test detection on a single frame."""
        # Create mock YOLO model and results
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock boxes
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=2)
        mock_boxes.cls = [MagicMock(item=lambda: 0), MagicMock(item=lambda: 0)]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 0.85)]

        # Mock xyxy tensors
        tensor1 = MagicMock()
        tensor1.cpu.return_value.numpy.return_value = np.array([10, 20, 110, 220])
        tensor2 = MagicMock()
        tensor2.cpu.return_value.numpy.return_value = np.array([200, 100, 300, 300])
        mock_boxes.xyxy = [tensor1, tensor2]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Run detection
        detector = YOLODetector()
        result = detector.detect(frame, frame_number=5)

        # Verify results
        assert isinstance(result, FrameDetections)
        assert result.frame_number == 5
        assert result.frame_width == 640
        assert result.frame_height == 480
        assert len(result.detections) == 2

        # Verify model was called correctly
        mock_model.predict.assert_called_once()

    @patch("app.ml.yolo_detector.YOLO")
    def test_detect_filters_classes(self, mock_yolo: MagicMock) -> None:
        """Test that detection filters to only valid classes."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock boxes with mixed classes: person (0), car (2), sports_ball (32)
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=3)
        mock_boxes.cls = [
            MagicMock(item=lambda: 0),  # person - valid
            MagicMock(item=lambda: 2),  # car - filtered out
            MagicMock(item=lambda: 32),  # sports_ball - valid
        ]
        mock_boxes.conf = [
            MagicMock(item=lambda: 0.9),
            MagicMock(item=lambda: 0.8),
            MagicMock(item=lambda: 0.75),
        ]

        tensor = MagicMock()
        tensor.cpu.return_value.numpy.return_value = np.array([10, 20, 110, 220])
        mock_boxes.xyxy = [tensor, tensor, tensor]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = YOLODetector()
        result = detector.detect(frame)

        # Should only have 2 detections (person and sports_ball)
        assert len(result.detections) == 2
        class_ids = {d.class_id for d in result.detections}
        assert class_ids == {0, 32}

    @patch("app.ml.yolo_detector.YOLO")
    def test_detect_empty_frame(self, mock_yolo: MagicMock) -> None:
        """Test detection on frame with no detections."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = YOLODetector()
        result = detector.detect(frame)

        assert len(result.detections) == 0

    @patch("app.ml.yolo_detector.YOLO")
    def test_detect_batch_empty_list(self, mock_yolo: MagicMock) -> None:
        """Test batch detection with empty frame list."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        detector = YOLODetector()
        result = detector.detect_batch([])

        assert result == []
        mock_model.predict.assert_not_called()

    @patch("app.ml.yolo_detector.YOLO")
    def test_detect_batch_multiple_frames(self, mock_yolo: MagicMock) -> None:
        """Test batch detection on multiple frames."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Create mock results for 3 frames
        mock_results = []
        for i in range(3):
            mock_boxes = MagicMock()
            mock_boxes.__len__ = MagicMock(return_value=1)
            mock_boxes.cls = [MagicMock(item=lambda: 0)]
            mock_boxes.conf = [MagicMock(item=lambda: 0.9)]
            tensor = MagicMock()
            tensor.cpu.return_value.numpy.return_value = np.array([10, 20, 110, 220])
            mock_boxes.xyxy = [tensor]

            mock_result = MagicMock()
            mock_result.boxes = mock_boxes
            mock_results.append(mock_result)

        mock_model.predict.return_value = mock_results

        # Create test frames
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

        detector = YOLODetector()
        results = detector.detect_batch(frames, start_frame_number=10)

        assert len(results) == 3
        assert results[0].frame_number == 10
        assert results[1].frame_number == 11
        assert results[2].frame_number == 12

    @patch("app.ml.yolo_detector.YOLO")
    def test_lazy_loading(self, mock_yolo: MagicMock) -> None:
        """Test that model is loaded lazily on first use."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = YOLODetector()

        # Model not loaded yet
        mock_yolo.assert_not_called()

        # First detection triggers model load
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)

        mock_yolo.assert_called_once_with("yolov8n.pt")

    @patch("app.ml.yolo_detector.YOLO")
    def test_model_loaded_once(self, mock_yolo: MagicMock) -> None:
        """Test that model is loaded only once."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = YOLODetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Multiple detections
        detector.detect(frame)
        detector.detect(frame)
        detector.detect(frame)

        # Model should be loaded only once
        mock_yolo.assert_called_once()


class TestYOLODetectorClassConstants:
    """Tests for YOLODetector class constants."""

    def test_valid_class_ids(self) -> None:
        """Test valid class IDs constant."""
        assert YOLODetector.PERSON_CLASS_ID == 0
        assert YOLODetector.SPORTS_BALL_CLASS_ID == 32
        assert YOLODetector.VALID_CLASS_IDS == {0, 32}

    def test_class_names(self) -> None:
        """Test class names mapping."""
        assert YOLODetector.CLASS_NAMES[0] == "person"
        assert YOLODetector.CLASS_NAMES[32] == "sports_ball"
