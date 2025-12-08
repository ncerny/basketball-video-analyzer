"""Video thumbnail generation service."""

import os
from pathlib import Path

import ffmpeg

from app.config import settings


class ThumbnailGeneratorService:
    """Service for generating video thumbnails."""

    def __init__(self, base_storage_path: str | None = None):
        """Initialize thumbnail generator service.

        Args:
            base_storage_path: Base directory for storing videos. Defaults to settings.video_storage_path.
        """
        self.base_path = Path(base_storage_path or settings.video_storage_path)

    def _get_thumbnail_storage_path(self, game_id: int) -> Path:
        """Get storage directory path for a game's thumbnails.

        Organizes thumbnails by game ID: {base_path}/game_{game_id}/thumbnails/

        Args:
            game_id: The game ID

        Returns:
            Path to the game's thumbnail storage directory
        """
        thumbnail_dir = self.base_path / f"game_{game_id}" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        return thumbnail_dir

    def _generate_thumbnail_filename(self, video_filename: str) -> str:
        """Generate thumbnail filename from video filename.

        Args:
            video_filename: Original video filename

        Returns:
            Thumbnail filename (replaces extension with .jpg)
        """
        return Path(video_filename).stem + "_thumb.jpg"

    def generate_thumbnail(
        self, video_relative_path: str, game_id: int, timestamp: float | None = None
    ) -> str:
        """Generate a thumbnail image from a video file.

        Extracts a frame from the video at the specified timestamp (or middle of video)
        and saves it as a JPEG thumbnail.

        Args:
            video_relative_path: Relative path to video file from base storage
            game_id: The game ID
            timestamp: Timestamp in seconds to extract frame from. If None, uses middle of video.

        Returns:
            Relative path to generated thumbnail from base storage

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If FFmpeg fails to generate thumbnail
        """
        video_path = self.base_path / video_relative_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get thumbnail storage directory
        thumbnail_dir = self._get_thumbnail_storage_path(game_id)

        # Generate thumbnail filename
        thumbnail_filename = self._generate_thumbnail_filename(video_path.name)
        thumbnail_path = thumbnail_dir / thumbnail_filename

        try:
            # Get video duration if timestamp not specified
            if timestamp is None:
                probe = ffmpeg.probe(str(video_path))
                duration = float(probe["format"].get("duration", 0))
                # Extract frame from middle of video
                timestamp = duration / 2

            # Extract frame using FFmpeg
            (
                ffmpeg.input(str(video_path), ss=timestamp)
                .filter("scale", 320, -1)  # Scale to 320px width, maintain aspect ratio
                .output(
                    str(thumbnail_path),
                    vframes=1,  # Extract 1 frame
                    format="image2",
                    vcodec="mjpeg",
                    **{"q:v": "2"},  # Quality (1-31, lower is better)
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            # Return relative path from base storage
            return str(thumbnail_path.relative_to(self.base_path))

        except ffmpeg.Error as e:
            # Include stderr output in error message
            stderr = e.stderr.decode() if e.stderr else "No error details"
            raise ValueError(f"FFmpeg error generating thumbnail: {stderr}")

    def delete_thumbnail(self, thumbnail_relative_path: str) -> None:
        """Delete a thumbnail file from storage.

        Args:
            thumbnail_relative_path: Relative path to thumbnail file from base storage

        Raises:
            FileNotFoundError: If thumbnail file doesn't exist
        """
        thumbnail_path = self.base_path / thumbnail_relative_path

        if not thumbnail_path.exists():
            raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_path}")

        thumbnail_path.unlink()

    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path for a thumbnail file.

        Args:
            relative_path: Relative path from base storage

        Returns:
            Absolute path to thumbnail file
        """
        return self.base_path / relative_path

    def thumbnail_exists(self, thumbnail_relative_path: str) -> bool:
        """Check if a thumbnail file exists.

        Args:
            thumbnail_relative_path: Relative path to thumbnail file from base storage

        Returns:
            True if thumbnail exists, False otherwise
        """
        thumbnail_path = self.base_path / thumbnail_relative_path
        return thumbnail_path.exists()
