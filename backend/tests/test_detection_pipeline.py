"""Tests for the detection pipeline service."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.types import BoundingBox, Detection, FrameDetections
from app.models.detection import PlayerDetection
from app.models.video import ProcessingStatus, Video
from app.services.detection_pipeline import (
    DetectionPipeline,
    DetectionPipelineConfig,
    DetectionPipelineResult,
)


class TestDetectionPipelineConfig:
    """Tests for DetectionPipelineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DetectionPipelineConfig()
        assert config.sample_interval == 3
        assert config.batch_size == 8
        assert config.confidence_threshold == 0.5
        assert config.device == "cpu"
        assert config.delete_existing is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DetectionPipelineConfig(
            sample_interval=5,
            batch_size=16,
            confidence_threshold=0.7,
            device="cuda",
            delete_existing=False,
        )
        assert config.sample_interval == 5
        assert config.batch_size == 16
        assert config.confidence_threshold == 0.7
        assert config.device == "cuda"
        assert config.delete_existing is False


class TestDetectionPipelineResult:
    """Tests for DetectionPipelineResult."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = DetectionPipelineResult(
            video_id=1,
            total_frames_processed=100,
            total_detections=50,
            persons_detected=45,
            balls_detected=5,
        )
        assert result.video_id == 1
        assert result.total_frames_processed == 100
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = DetectionPipelineResult(
            video_id=1,
            total_frames_processed=0,
            total_detections=0,
            persons_detected=0,
            balls_detected=0,
            error="Video not found",
        )
        assert result.error == "Video not found"


class TestDetectionPipeline:
    """Tests for DetectionPipeline."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = MagicMock(spec=AsyncSession)
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.add = MagicMock()
        return db

    @pytest.fixture
    def mock_video(self, tmp_path):
        """Create a mock video with a temp file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")

        video = MagicMock(spec=Video)
        video.id = 1
        video.file_path = str(video_file.name)
        video.processing_status = ProcessingStatus.PENDING
        video.processed = False
        return video

    @pytest.fixture
    def mock_frame_detections(self):
        """Create mock frame detections."""

        def create_detections(frame_number: int) -> FrameDetections:
            return FrameDetections(
                frame_number=frame_number,
                detections=[
                    Detection(
                        bbox=BoundingBox(x=100, y=100, width=50, height=100),
                        confidence=0.9,
                        class_id=0,  # person
                        class_name="person",
                    ),
                    Detection(
                        bbox=BoundingBox(x=200, y=200, width=30, height=30),
                        confidence=0.8,
                        class_id=32,  # sports_ball
                        class_name="sports_ball",
                    ),
                ],
                frame_width=640,
                frame_height=480,
            )

        return create_detections

    @pytest.mark.asyncio
    async def test_process_video_not_found(self, mock_db):
        """Test processing when video not found."""
        # Setup mock to return no video
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        pipeline = DetectionPipeline(mock_db)
        result = await pipeline.process_video(999)

        assert result.error == "Video not found: 999"
        assert result.total_frames_processed == 0

    @pytest.mark.asyncio
    async def test_process_video_file_not_found(self, mock_db, mock_video):
        """Test processing when video file doesn't exist."""
        # Point to non-existent file
        mock_video.file_path = "nonexistent.mp4"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_video
        mock_db.execute.return_value = mock_result

        with patch("app.services.detection_pipeline.settings") as mock_settings:
            mock_settings.video_storage_path = "/tmp"
            mock_settings.yolo_confidence_threshold = 0.5
            mock_settings.ml_device = "cpu"
            mock_settings.yolo_model_name = "yolov8n.pt"
            mock_settings.yolo_batch_size_cpu = 8
            mock_settings.yolo_batch_size_mps = 16
            mock_settings.yolo_batch_size_cuda = 32
            mock_settings.enable_inference_timing = False
            pipeline = DetectionPipeline(mock_db)
            result = await pipeline.process_video(1)

        assert result.error is not None and "Video file not found" in result.error
        assert mock_video.processing_status == ProcessingStatus.FAILED

    @pytest.mark.asyncio
    async def test_process_video_success(
        self, mock_db, mock_video, mock_frame_detections, tmp_path
    ):
        """Test successful video processing."""
        # Setup video file
        video_file = tmp_path / mock_video.file_path
        video_file.write_bytes(b"fake video")

        # Setup mock database
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_video
        mock_db.execute.return_value = mock_result

        # Mock frame extractor
        mock_extracted_frame = MagicMock()
        mock_extracted_frame.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_extracted_frame.frame_number = 0

        mock_metadata = MagicMock()
        mock_metadata.total_frames = 30
        mock_metadata.fps = 30.0

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [mock_frame_detections(0)]

        with (
            patch("app.services.detection_pipeline.FrameExtractor") as MockExtractor,
            patch.object(DetectionPipeline, "_get_detector", return_value=mock_detector),
            patch("app.services.detection_pipeline.settings") as mock_settings,
        ):
            mock_settings.video_storage_path = str(tmp_path)
            mock_settings.yolo_confidence_threshold = 0.5
            mock_settings.ml_device = "cpu"
            mock_settings.yolo_model_name = "yolov8n.pt"
            mock_settings.yolo_batch_size_cpu = 8
            mock_settings.yolo_batch_size_mps = 16
            mock_settings.yolo_batch_size_cuda = 32
            mock_settings.enable_inference_timing = False
            # Setup extractor mock
            mock_extractor_instance = MagicMock()
            mock_extractor_instance.get_metadata.return_value = mock_metadata
            mock_extractor_instance.count_sampled_frames.return_value = 1
            mock_extractor_instance.extract_frames_sampled.return_value = iter(
                [mock_extracted_frame]
            )
            mock_extractor_instance.__enter__ = MagicMock(return_value=mock_extractor_instance)
            mock_extractor_instance.__exit__ = MagicMock(return_value=None)
            MockExtractor.return_value = mock_extractor_instance

            pipeline = DetectionPipeline(mock_db)
            result = await pipeline.process_video(1)

        assert result.error is None
        assert result.video_id == 1
        assert result.total_frames_processed == 1
        assert result.total_detections == 2
        assert result.persons_detected == 1
        assert result.balls_detected == 1
        assert mock_video.processing_status == ProcessingStatus.COMPLETED
        assert mock_video.processed is True

    @pytest.mark.asyncio
    async def test_process_video_with_progress_callback(self, mock_db, mock_video, tmp_path):
        """Test progress callback is invoked during processing."""
        video_file = tmp_path / mock_video.file_path
        video_file.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_video
        mock_db.execute.return_value = mock_result

        progress_updates = []

        def track_progress(current, total, message):
            progress_updates.append((current, total, message))

        mock_metadata = MagicMock()
        mock_metadata.total_frames = 30

        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [FrameDetections(frame_number=0, detections=[])]

        mock_extracted_frame = MagicMock()
        mock_extracted_frame.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_extracted_frame.frame_number = 0

        with (
            patch("app.services.detection_pipeline.FrameExtractor") as MockExtractor,
            patch.object(DetectionPipeline, "_get_detector", return_value=mock_detector),
            patch("app.services.detection_pipeline.settings") as mock_settings,
        ):
            mock_settings.video_storage_path = str(tmp_path)
            mock_settings.yolo_confidence_threshold = 0.5
            mock_settings.ml_device = "cpu"
            mock_settings.yolo_model_name = "yolov8n.pt"
            mock_settings.yolo_batch_size_cpu = 8
            mock_settings.yolo_batch_size_mps = 16
            mock_settings.yolo_batch_size_cuda = 32
            mock_settings.enable_inference_timing = False
            mock_metadata.fps = 30.0
            mock_extractor = MagicMock()
            mock_extractor.get_metadata.return_value = mock_metadata
            mock_extractor.count_sampled_frames.return_value = 1
            mock_extractor.extract_frames_sampled.return_value = iter([mock_extracted_frame])
            mock_extractor.__enter__ = MagicMock(return_value=mock_extractor)
            mock_extractor.__exit__ = MagicMock(return_value=None)
            MockExtractor.return_value = mock_extractor

            pipeline = DetectionPipeline(mock_db)
            await pipeline.process_video(1, progress_callback=track_progress)

        # Should have multiple progress updates
        assert len(progress_updates) > 0
        # Should end at 100%
        assert progress_updates[-1][0] == 100

    @pytest.mark.asyncio
    async def test_delete_existing_detections(self, mock_db, mock_video, tmp_path):
        """Test that existing detections are deleted."""
        video_file = tmp_path / mock_video.file_path
        video_file.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_video
        mock_db.execute.return_value = mock_result

        mock_metadata = MagicMock()
        mock_metadata.total_frames = 30

        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = []

        with (
            patch("app.services.detection_pipeline.FrameExtractor") as MockExtractor,
            patch.object(DetectionPipeline, "_get_detector", return_value=mock_detector),
            patch("app.services.detection_pipeline.settings") as mock_settings,
        ):
            mock_settings.video_storage_path = str(tmp_path)
            mock_settings.yolo_confidence_threshold = 0.5
            mock_settings.ml_device = "cpu"
            mock_settings.yolo_model_name = "yolov8n.pt"
            mock_settings.yolo_batch_size_cpu = 8
            mock_settings.yolo_batch_size_mps = 16
            mock_settings.yolo_batch_size_cuda = 32
            mock_settings.enable_inference_timing = False
            mock_metadata.fps = 30.0
            mock_extractor = MagicMock()
            mock_extractor.get_metadata.return_value = mock_metadata
            mock_extractor.count_sampled_frames.return_value = 0
            mock_extractor.extract_frames_sampled.return_value = iter([])
            mock_extractor.__enter__ = MagicMock(return_value=mock_extractor)
            mock_extractor.__exit__ = MagicMock(return_value=None)
            MockExtractor.return_value = mock_extractor

            config = DetectionPipelineConfig(delete_existing=True)
            pipeline = DetectionPipeline(mock_db, config)
            await pipeline.process_video(1)

        # Verify delete was called (first execute after getting video is for delete)
        assert mock_db.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_db, mock_video, tmp_path):
        """Test exception handling during processing."""
        video_file = tmp_path / mock_video.file_path
        video_file.write_bytes(b"fake video")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_video
        mock_db.execute.return_value = mock_result

        with (
            patch("app.services.detection_pipeline.FrameExtractor") as MockExtractor,
            patch("app.services.detection_pipeline.settings") as mock_settings,
        ):
            mock_settings.video_storage_path = str(tmp_path)
            mock_settings.yolo_confidence_threshold = 0.5
            mock_settings.ml_device = "cpu"
            mock_settings.yolo_model_name = "yolov8n.pt"
            mock_settings.yolo_batch_size_cpu = 8
            mock_settings.yolo_batch_size_mps = 16
            mock_settings.yolo_batch_size_cuda = 32
            mock_settings.enable_inference_timing = False
            MockExtractor.side_effect = RuntimeError("Extractor failed")

            pipeline = DetectionPipeline(mock_db)
            result = await pipeline.process_video(1)

        assert result.error is not None and "Extractor failed" in result.error
        assert mock_video.processing_status == ProcessingStatus.FAILED

    def test_resolve_device_auto_cpu(self):
        """Test device resolution falls back to CPU."""
        with patch.dict("sys.modules", {"torch": None}):
            device = DetectionPipeline._resolve_device("auto")
            assert device == "cpu"

    def test_resolve_device_explicit(self):
        """Test explicit device setting is preserved."""
        assert DetectionPipeline._resolve_device("cuda") == "cuda"
        assert DetectionPipeline._resolve_device("mps") == "mps"
        assert DetectionPipeline._resolve_device("cpu") == "cpu"


class TestStoreFrameDetections:
    """Tests for storing frame detections."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = MagicMock(spec=AsyncSession)
        db.add = MagicMock()
        return db

    @pytest.mark.asyncio
    async def test_store_detections(self, mock_db):
        """Test storing detections in database."""
        frame_detections = FrameDetections(
            frame_number=10,
            detections=[
                Detection(
                    bbox=BoundingBox(x=100, y=100, width=50, height=100),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                ),
                Detection(
                    bbox=BoundingBox(x=200, y=200, width=30, height=30),
                    confidence=0.8,
                    class_id=32,
                    class_name="sports_ball",
                ),
            ],
        )

        pipeline = DetectionPipeline(mock_db)
        stats = await pipeline._store_frame_detections(1, 10, frame_detections)

        assert stats["total"] == 2
        assert stats["persons"] == 1
        assert stats["balls"] == 1
        assert mock_db.add.call_count == 2

    @pytest.mark.asyncio
    async def test_store_empty_detections(self, mock_db):
        """Test storing empty frame detections."""
        frame_detections = FrameDetections(frame_number=10, detections=[])

        pipeline = DetectionPipeline(mock_db)
        stats = await pipeline._store_frame_detections(1, 10, frame_detections)

        assert stats["total"] == 0
        assert mock_db.add.call_count == 0


class TestOCRIntegration:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=AsyncSession)
        db.add = MagicMock()
        return db

    def test_ocr_sampling_logic(self, mock_db):
        config = DetectionPipelineConfig(ocr_sample_rate=5)
        pipeline = DetectionPipeline(mock_db, config)

        results = []
        for _ in range(15):
            results.append(pipeline._should_run_ocr_for_track(1))

        assert results == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]

    def test_ocr_sampling_per_track(self, mock_db):
        config = DetectionPipelineConfig(ocr_sample_rate=3)
        pipeline = DetectionPipeline(mock_db, config)

        track1_results = [pipeline._should_run_ocr_for_track(1) for _ in range(6)]
        track2_results = [pipeline._should_run_ocr_for_track(2) for _ in range(6)]

        assert track1_results == [True, False, False, True, False, False]
        assert track2_results == [True, False, False, True, False, False]

    def test_reset_ocr_state(self, mock_db):
        config = DetectionPipelineConfig(ocr_sample_rate=3)
        pipeline = DetectionPipeline(mock_db, config)

        pipeline._should_run_ocr_for_track(1)
        pipeline._should_run_ocr_for_track(1)
        assert pipeline._should_run_ocr_for_track(1) is False

        pipeline._reset_ocr_state()
        assert pipeline._should_run_ocr_for_track(1) is True

    @pytest.mark.asyncio
    async def test_run_ocr_disabled(self, mock_db):
        config = DetectionPipelineConfig(enable_jersey_ocr=False)
        pipeline = DetectionPipeline(mock_db, config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = FrameDetections(
            frame_number=0,
            detections=[
                Detection(
                    bbox=BoundingBox(x=100, y=100, width=80, height=200),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                    tracking_id=1,
                )
            ],
        )

        count = await pipeline._run_ocr_on_frame(1, 0, frame, detections)
        assert count == 0
        assert mock_db.add.call_count == 0

    @pytest.mark.asyncio
    async def test_run_ocr_skips_non_person(self, mock_db):
        config = DetectionPipelineConfig(enable_jersey_ocr=True)
        pipeline = DetectionPipeline(mock_db, config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = FrameDetections(
            frame_number=0,
            detections=[
                Detection(
                    bbox=BoundingBox(x=100, y=100, width=30, height=30),
                    confidence=0.9,
                    class_id=32,
                    class_name="sports_ball",
                    tracking_id=1,
                )
            ],
        )

        count = await pipeline._run_ocr_on_frame(1, 0, frame, detections)
        assert count == 0

    @pytest.mark.asyncio
    async def test_run_ocr_skips_no_tracking_id(self, mock_db):
        config = DetectionPipelineConfig(enable_jersey_ocr=True)
        pipeline = DetectionPipeline(mock_db, config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = FrameDetections(
            frame_number=0,
            detections=[
                Detection(
                    bbox=BoundingBox(x=100, y=100, width=80, height=200),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                    tracking_id=None,
                )
            ],
        )

        count = await pipeline._run_ocr_on_frame(1, 0, frame, detections)
        assert count == 0
