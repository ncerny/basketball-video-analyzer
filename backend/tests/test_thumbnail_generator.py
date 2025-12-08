"""Tests for thumbnail generator service."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.thumbnail_generator import ThumbnailGeneratorService


@pytest.fixture
def thumbnail_service(tmp_path: Path) -> ThumbnailGeneratorService:
    """Create thumbnail service with temporary storage."""
    return ThumbnailGeneratorService(base_storage_path=str(tmp_path))


def test_get_thumbnail_storage_path(thumbnail_service: ThumbnailGeneratorService, tmp_path: Path) -> None:
    """Test thumbnail storage path generation."""
    game_id = 1
    thumbnail_dir = thumbnail_service._get_thumbnail_storage_path(game_id)

    assert thumbnail_dir == tmp_path / "game_1" / "thumbnails"
    assert thumbnail_dir.exists()


def test_generate_thumbnail_filename(thumbnail_service: ThumbnailGeneratorService) -> None:
    """Test thumbnail filename generation."""
    video_filename = "game1_20231215_120000_quarter1.mp4"
    thumbnail_filename = thumbnail_service._generate_thumbnail_filename(video_filename)

    assert thumbnail_filename == "game1_20231215_120000_quarter1_thumb.jpg"


def test_thumbnail_exists(thumbnail_service: ThumbnailGeneratorService, tmp_path: Path) -> None:
    """Test thumbnail existence check."""
    # Create a dummy thumbnail file
    thumbnail_dir = tmp_path / "game_1" / "thumbnails"
    thumbnail_dir.mkdir(parents=True)
    thumbnail_path = thumbnail_dir / "test_thumb.jpg"
    thumbnail_path.touch()

    relative_path = str(thumbnail_path.relative_to(tmp_path))

    assert thumbnail_service.thumbnail_exists(relative_path) is True
    assert thumbnail_service.thumbnail_exists("nonexistent/path.jpg") is False


def test_get_absolute_path(thumbnail_service: ThumbnailGeneratorService, tmp_path: Path) -> None:
    """Test absolute path resolution."""
    relative_path = "game_1/thumbnails/test_thumb.jpg"
    absolute_path = thumbnail_service.get_absolute_path(relative_path)

    assert absolute_path == tmp_path / relative_path


def test_delete_thumbnail(thumbnail_service: ThumbnailGeneratorService, tmp_path: Path) -> None:
    """Test thumbnail deletion."""
    # Create a dummy thumbnail file
    thumbnail_dir = tmp_path / "game_1" / "thumbnails"
    thumbnail_dir.mkdir(parents=True)
    thumbnail_path = thumbnail_dir / "test_thumb.jpg"
    thumbnail_path.touch()

    relative_path = str(thumbnail_path.relative_to(tmp_path))

    # Delete thumbnail
    thumbnail_service.delete_thumbnail(relative_path)

    assert not thumbnail_path.exists()


def test_delete_thumbnail_not_found(thumbnail_service: ThumbnailGeneratorService) -> None:
    """Test deleting non-existent thumbnail raises error."""
    with pytest.raises(FileNotFoundError):
        thumbnail_service.delete_thumbnail("nonexistent/path.jpg")


@patch("app.services.thumbnail_generator.ffmpeg")
def test_generate_thumbnail_with_timestamp(
    mock_ffmpeg: MagicMock,
    thumbnail_service: ThumbnailGeneratorService,
    tmp_path: Path,
) -> None:
    """Test thumbnail generation with specific timestamp."""
    # Create a dummy video file
    video_dir = tmp_path / "game_1"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "test_video.mp4"
    video_path.touch()

    relative_video_path = str(video_path.relative_to(tmp_path))

    # Mock FFmpeg
    mock_input = MagicMock()
    mock_filter = MagicMock()
    mock_output = MagicMock()
    mock_overwrite = MagicMock()

    mock_ffmpeg.input.return_value = mock_input
    mock_input.filter.return_value = mock_filter
    mock_filter.output.return_value = mock_output
    mock_output.overwrite_output.return_value = mock_overwrite
    mock_overwrite.run.return_value = None

    # Generate thumbnail
    thumbnail_path = thumbnail_service.generate_thumbnail(
        relative_video_path, game_id=1, timestamp=30.0
    )

    # Verify FFmpeg was called correctly
    mock_ffmpeg.input.assert_called_once_with(str(video_path), ss=30.0)
    mock_input.filter.assert_called_once_with("scale", 320, -1)

    # Verify thumbnail path
    assert thumbnail_path == "game_1/thumbnails/test_video_thumb.jpg"


@patch("app.services.thumbnail_generator.ffmpeg")
def test_generate_thumbnail_without_timestamp(
    mock_ffmpeg: MagicMock,
    thumbnail_service: ThumbnailGeneratorService,
    tmp_path: Path,
) -> None:
    """Test thumbnail generation without timestamp (uses middle of video)."""
    # Create a dummy video file
    video_dir = tmp_path / "game_1"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "test_video.mp4"
    video_path.touch()

    relative_video_path = str(video_path.relative_to(tmp_path))

    # Mock FFmpeg probe
    mock_probe_result = {
        "format": {"duration": "100.0"},
        "streams": [{"codec_type": "video"}],
    }
    mock_ffmpeg.probe.return_value = mock_probe_result

    # Mock FFmpeg input/filter/output
    mock_input = MagicMock()
    mock_filter = MagicMock()
    mock_output = MagicMock()
    mock_overwrite = MagicMock()

    mock_ffmpeg.input.return_value = mock_input
    mock_input.filter.return_value = mock_filter
    mock_filter.output.return_value = mock_output
    mock_output.overwrite_output.return_value = mock_overwrite
    mock_overwrite.run.return_value = None

    # Generate thumbnail
    thumbnail_path = thumbnail_service.generate_thumbnail(
        relative_video_path, game_id=1, timestamp=None
    )

    # Verify FFmpeg probe was called
    mock_ffmpeg.probe.assert_called_once_with(str(video_path))

    # Verify FFmpeg input was called with middle timestamp (100 / 2 = 50)
    mock_ffmpeg.input.assert_called_once_with(str(video_path), ss=50.0)

    # Verify thumbnail path
    assert thumbnail_path == "game_1/thumbnails/test_video_thumb.jpg"


def test_generate_thumbnail_video_not_found(thumbnail_service: ThumbnailGeneratorService) -> None:
    """Test generating thumbnail for non-existent video raises error."""
    with pytest.raises(FileNotFoundError):
        thumbnail_service.generate_thumbnail("nonexistent/video.mp4", game_id=1)
