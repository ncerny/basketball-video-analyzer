"""Tests for the frame extraction service."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.frame_extractor import (
    ExtractedFrame,
    FrameExtractionError,
    FrameExtractor,
    VideoMetadata,
)


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_frame_duration_ms(self):
        """Test frame duration calculation."""
        metadata = VideoMetadata(
            total_frames=300,
            fps=30.0,
            width=1920,
            height=1080,
            duration_seconds=10.0,
        )
        assert metadata.frame_duration_ms == pytest.approx(33.333, rel=0.01)

    def test_frame_duration_ms_zero_fps(self):
        """Test frame duration with zero FPS returns 0."""
        metadata = VideoMetadata(
            total_frames=0,
            fps=0.0,
            width=0,
            height=0,
            duration_seconds=0.0,
        )
        assert metadata.frame_duration_ms == 0.0


class TestFrameExtractor:
    """Tests for FrameExtractor service."""

    @pytest.fixture
    def mock_video_capture(self):
        """Create a mock VideoCapture object."""
        with patch("cv2.VideoCapture") as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = self._mock_cap_get
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap_class.return_value = mock_cap
            yield mock_cap

    def _mock_cap_get(self, prop_id):
        """Mock VideoCapture.get() method."""
        import cv2

        props = {
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }
        return props.get(prop_id, 0)

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary file that pretends to be a video."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            path = Path(f.name)
        yield path
        path.unlink(missing_ok=True)

    def test_init_file_not_found(self):
        """Test initialization fails for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            FrameExtractor("/nonexistent/video.mp4")

    def test_init_with_valid_path(self, temp_video_file):
        """Test initialization succeeds for existing file."""
        extractor = FrameExtractor(temp_video_file)
        assert extractor._video_path == temp_video_file

    def test_context_manager(self, temp_video_file, mock_video_capture):
        """Test context manager opens and closes video."""
        with FrameExtractor(temp_video_file) as extractor:
            assert extractor._cap is not None
            mock_video_capture.isOpened.assert_called()
        # After context exit, release should be called
        mock_video_capture.release.assert_called_once()

    def test_get_metadata(self, temp_video_file, mock_video_capture):
        """Test metadata extraction."""
        extractor = FrameExtractor(temp_video_file)
        metadata = extractor.get_metadata()

        assert metadata.total_frames == 300
        assert metadata.fps == 30.0
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.duration_seconds == 10.0
        extractor.close()

    def test_get_metadata_cached(self, temp_video_file, mock_video_capture):
        """Test metadata is cached after first call."""
        extractor = FrameExtractor(temp_video_file)

        metadata1 = extractor.get_metadata()
        metadata2 = extractor.get_metadata()

        assert metadata1 is metadata2
        extractor.close()

    def test_extract_frame(self, temp_video_file, mock_video_capture):
        """Test extracting a single frame."""
        with FrameExtractor(temp_video_file) as extractor:
            result = extractor.extract_frame(10)

            assert isinstance(result, ExtractedFrame)
            assert result.frame_number == 10
            assert result.frame.shape == (480, 640, 3)
            mock_video_capture.set.assert_called()

    def test_extract_frame_out_of_range(self, temp_video_file, mock_video_capture):
        """Test extracting frame beyond video length."""
        with FrameExtractor(temp_video_file) as extractor:
            with pytest.raises(ValueError, match="out of range"):
                extractor.extract_frame(500)

    def test_extract_frame_negative(self, temp_video_file, mock_video_capture):
        """Test extracting negative frame number."""
        with FrameExtractor(temp_video_file) as extractor:
            with pytest.raises(ValueError, match="out of range"):
                extractor.extract_frame(-1)

    def test_extract_frame_read_failure(self, temp_video_file, mock_video_capture):
        """Test handling read failure."""
        mock_video_capture.read.return_value = (False, None)

        with FrameExtractor(temp_video_file) as extractor:
            with pytest.raises(FrameExtractionError, match="Failed to read frame"):
                extractor.extract_frame(10)

    def test_extract_frame_at_timestamp(self, temp_video_file, mock_video_capture):
        """Test extracting frame at specific timestamp."""
        with FrameExtractor(temp_video_file) as extractor:
            # At 30fps, frame_duration = 33.33ms
            # Use 330ms for clean calculation: 330 / 33.33 = ~9.9 -> frame 9
            # Frame 9 covers 300-333.33ms, so 330ms falls in frame 9
            result = extractor.extract_frame_at_timestamp(330.0)

            assert result.frame_number == 9

    def test_extract_frame_at_timestamp_out_of_range(self, temp_video_file, mock_video_capture):
        """Test timestamp beyond video duration."""
        with FrameExtractor(temp_video_file) as extractor:
            with pytest.raises(ValueError, match="out of range"):
                extractor.extract_frame_at_timestamp(20000.0)  # 20 seconds on 10 second video

    def test_extract_frames_sampled(self, temp_video_file, mock_video_capture):
        """Test sampled frame extraction."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = list(extractor.extract_frames_sampled(sample_interval=3, end_frame=15))

            # Frames 0, 3, 6, 9, 12 = 5 frames
            assert len(frames) == 5
            assert frames[0].frame_number == 0
            assert frames[1].frame_number == 3
            assert frames[4].frame_number == 12

    def test_extract_frames_sampled_with_start(self, temp_video_file, mock_video_capture):
        """Test sampled extraction with start offset."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = list(
                extractor.extract_frames_sampled(sample_interval=3, start_frame=6, end_frame=15)
            )

            # Frames 6, 9, 12 = 3 frames
            assert len(frames) == 3
            assert frames[0].frame_number == 6

    def test_extract_frames_sampled_invalid_interval(self, temp_video_file, mock_video_capture):
        """Test invalid sample interval."""
        with FrameExtractor(temp_video_file) as extractor:
            with pytest.raises(ValueError, match="sample_interval must be at least 1"):
                list(extractor.extract_frames_sampled(sample_interval=0))

    def test_extract_frames_sampled_skips_failed_reads(self, temp_video_file, mock_video_capture):
        """Test that failed reads are skipped, not raised."""
        read_results = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 2
        read_results.insert(1, (False, None))  # Middle read fails
        mock_video_capture.read.side_effect = read_results

        with FrameExtractor(temp_video_file) as extractor:
            frames = list(extractor.extract_frames_sampled(sample_interval=3, end_frame=9))

        # Should have 2 frames (first and third succeeded)
        assert len(frames) == 2

    def test_extract_frames_batch(self, temp_video_file, mock_video_capture):
        """Test batch frame extraction."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = extractor.extract_frames_batch([5, 10, 15])

            assert len(frames) == 3

    def test_extract_frames_batch_empty(self, temp_video_file, mock_video_capture):
        """Test batch extraction with empty list."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = extractor.extract_frames_batch([])

            assert frames == []

    def test_extract_frames_batch_filters_invalid(self, temp_video_file, mock_video_capture):
        """Test batch extraction filters out invalid frame numbers."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = extractor.extract_frames_batch([-1, 5, 500])

            # Only frame 5 is valid
            assert len(frames) == 1
            assert frames[0].frame_number == 5

    def test_extract_frames_batch_deduplicates(self, temp_video_file, mock_video_capture):
        """Test batch extraction deduplicates frame numbers."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = extractor.extract_frames_batch([5, 5, 5, 10, 10])

            # Should only extract unique frames
            assert len(frames) == 2

    def test_count_sampled_frames(self, temp_video_file, mock_video_capture):
        """Test counting sampled frames."""
        with FrameExtractor(temp_video_file) as extractor:
            count = extractor.count_sampled_frames(sample_interval=3)

            # 300 frames / 3 = 100 frames
            assert count == 100

    def test_count_sampled_frames_with_range(self, temp_video_file, mock_video_capture):
        """Test counting with start/end range."""
        with FrameExtractor(temp_video_file) as extractor:
            count = extractor.count_sampled_frames(sample_interval=3, start_frame=0, end_frame=15)

            # Frames 0, 3, 6, 9, 12 = 5
            assert count == 5

    def test_count_sampled_frames_empty_range(self, temp_video_file, mock_video_capture):
        """Test counting with invalid range."""
        with FrameExtractor(temp_video_file) as extractor:
            count = extractor.count_sampled_frames(sample_interval=3, start_frame=15, end_frame=10)

            assert count == 0

    def test_extract_frames_range(self, temp_video_file, mock_video_capture):
        """Test extracting all frames in a range."""
        with FrameExtractor(temp_video_file) as extractor:
            frames = list(extractor.extract_frames_range(start_frame=0, end_frame=5))

            assert len(frames) == 5
            assert frames[0].frame_number == 0
            assert frames[4].frame_number == 4

    def test_close_releases_capture(self, temp_video_file, mock_video_capture):
        """Test close() releases VideoCapture."""
        extractor = FrameExtractor(temp_video_file)
        extractor.get_metadata()  # Opens video
        extractor.close()

        mock_video_capture.release.assert_called_once()

    def test_close_idempotent(self, temp_video_file, mock_video_capture):
        """Test close() can be called multiple times."""
        extractor = FrameExtractor(temp_video_file)
        extractor.get_metadata()
        extractor.close()
        extractor.close()  # Should not raise

        # Release only called once
        mock_video_capture.release.assert_called_once()


class TestFrameExtractorOpenVideoFailure:
    """Tests for video open failure scenarios."""

    def test_open_video_failure(self, tmp_path):
        """Test handling when VideoCapture fails to open."""
        video_file = tmp_path / "fake.mp4"
        video_file.write_bytes(b"not a video")

        with patch("cv2.VideoCapture") as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap

            extractor = FrameExtractor(video_file)

            with pytest.raises(FrameExtractionError, match="Failed to open video"):
                extractor.get_metadata()
