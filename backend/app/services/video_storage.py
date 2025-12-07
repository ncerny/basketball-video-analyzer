"""Video storage and metadata extraction service."""

import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any

import ffmpeg
from fastapi import UploadFile

from app.config import settings


class VideoStorageService:
    """Service for handling video file uploads and metadata extraction."""

    def __init__(self, base_storage_path: str | None = None):
        """Initialize video storage service.

        Args:
            base_storage_path: Base directory for storing videos. Defaults to settings.video_storage_path.
        """
        self.base_path = Path(base_storage_path or settings.video_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_video_storage_path(self, game_id: int) -> Path:
        """Get storage directory path for a game's videos.

        Organizes videos by game ID: {base_path}/game_{game_id}/

        Args:
            game_id: The game ID

        Returns:
            Path to the game's video storage directory
        """
        game_dir = self.base_path / f"game_{game_id}"
        game_dir.mkdir(parents=True, exist_ok=True)
        return game_dir

    def _generate_filename(self, original_filename: str, game_id: int) -> str:
        """Generate a unique filename for the uploaded video.

        Format: game{game_id}_{timestamp}_{original_name}

        Args:
            original_filename: Original filename from upload
            game_id: The game ID

        Returns:
            Generated unique filename
        """
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        # Sanitize original filename (keep extension)
        name_parts = Path(original_filename).stem
        extension = Path(original_filename).suffix
        safe_name = "".join(c for c in name_parts if c.isalnum() or c in ("_", "-"))[:50]
        return f"game{game_id}_{timestamp}_{safe_name}{extension}"

    async def save_video(self, file: UploadFile, game_id: int) -> str:
        """Save uploaded video file to storage.

        Args:
            file: The uploaded file
            game_id: The game ID this video belongs to

        Returns:
            Relative file path from base storage directory
        """
        storage_dir = self._get_video_storage_path(game_id)
        filename = self._generate_filename(file.filename or "video.mp4", game_id)
        file_path = storage_dir / filename

        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Return relative path from base storage
        return str(file_path.relative_to(self.base_path))

    def extract_metadata(self, relative_path: str) -> dict[str, Any]:
        """Extract video metadata using FFmpeg.

        Args:
            relative_path: Relative path to video file from base storage

        Returns:
            Dictionary containing video metadata:
            - duration_seconds: Video duration in seconds
            - fps: Frames per second
            - resolution: Resolution string (e.g., "1920x1080")
            - recorded_at: Recording timestamp if available in metadata

        Raises:
            ffmpeg.Error: If FFmpeg fails to probe the video
            FileNotFoundError: If video file doesn't exist
        """
        file_path = self.base_path / relative_path

        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        try:
            # Probe video file
            probe = ffmpeg.probe(str(file_path))

            # Extract video stream info
            video_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
            )

            if not video_stream:
                raise ValueError("No video stream found in file")

            # Extract metadata
            duration = float(probe["format"].get("duration", 0))

            # Calculate FPS from avg_frame_rate (format: "num/den")
            fps_str = video_stream.get("avg_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)

            # Get resolution
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            resolution = f"{width}x{height}"

            # Try to extract recording timestamp from metadata
            recorded_at = None
            creation_time = probe["format"].get("tags", {}).get("creation_time")
            if creation_time:
                try:
                    # Parse ISO 8601 timestamp
                    recorded_at = dt.datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            return {
                "duration_seconds": duration,
                "fps": fps,
                "resolution": resolution,
                "recorded_at": recorded_at,
            }

        except ffmpeg.Error as e:
            # Include stderr output in error message
            stderr = e.stderr.decode() if e.stderr else "No error details"
            raise ValueError(f"FFmpeg error: {stderr}")

    def delete_video(self, relative_path: str) -> None:
        """Delete a video file from storage.

        Args:
            relative_path: Relative path to video file from base storage

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        file_path = self.base_path / relative_path

        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        file_path.unlink()

    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path for a video file.

        Args:
            relative_path: Relative path from base storage

        Returns:
            Absolute path to video file
        """
        return self.base_path / relative_path
