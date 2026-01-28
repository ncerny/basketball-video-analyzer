"""Tests for SAM3 frame extractor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSAM3FrameExtractor:
    """Tests for SAM3 frame extraction utility."""

    def test_extract_frames_creates_directory(self, tmp_path: Path) -> None:
        """Test that extract_frames creates output directory."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"

        # Mock video capture to return empty
        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.read.return_value = (False, None)
            mock_cap.return_value.get.return_value = 30.0  # fps

            extractor.extract_frames(Path("fake_video.mp4"), output_dir)

        assert output_dir.exists()

    def test_extract_frames_respects_sample_interval(self, tmp_path: Path) -> None:
        """Test that extract_frames samples every Nth frame."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"

        # Create fake frames
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.return_value = 30.0

            # Return 10 frames then stop
            call_count = [0]
            def mock_read():
                call_count[0] += 1
                if call_count[0] <= 10:
                    return (True, fake_frame.copy())
                return (False, None)

            mock_cap.return_value.read.side_effect = mock_read

            with patch("app.ml.sam3_frame_extractor.cv2.imwrite") as mock_write:
                mock_write.return_value = True
                result = extractor.extract_frames(
                    Path("fake.mp4"), output_dir, sample_interval=3
                )

        # With 10 frames and interval=3, should get frames 0, 3, 6, 9 = 4 frames
        assert result.frame_count == 4
        assert result.sample_interval == 3

    def test_extract_frames_returns_metadata(self, tmp_path: Path) -> None:
        """Test that extract_frames returns correct metadata."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = lambda prop: {
                5: 30.0,  # CAP_PROP_FPS
                3: 1280,  # CAP_PROP_FRAME_WIDTH
                4: 720,   # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)

            call_count = [0]
            def mock_read():
                call_count[0] += 1
                if call_count[0] <= 3:
                    return (True, fake_frame.copy())
                return (False, None)

            mock_cap.return_value.read.side_effect = mock_read

            with patch("app.ml.sam3_frame_extractor.cv2.imwrite", return_value=True):
                result = extractor.extract_frames(Path("fake.mp4"), output_dir)

        assert result.fps == 30.0
        assert result.width == 1280
        assert result.height == 720
        assert result.output_dir == output_dir

    def test_context_manager_cleanup(self, tmp_path: Path) -> None:
        """Test that context manager cleans up temp directory."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()

        with patch("app.ml.sam3_frame_extractor.settings") as mock_settings:
            mock_settings.sam3_temp_frames_dir = tmp_path

            with extractor.temp_frame_folder("test_video") as folder:
                temp_path = folder
                assert folder.exists()
                # Create a file to verify cleanup
                (folder / "test.jpg").touch()

            # After context, folder should be deleted
            assert not temp_path.exists()
