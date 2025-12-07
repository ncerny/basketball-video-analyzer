"""Video schemas for API requests and responses."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class VideoBase(BaseModel):
    """Base video schema with common fields."""

    game_id: int = Field(..., ge=1, description="ID of the game this video belongs to")
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to video file")
    duration_seconds: float = Field(..., gt=0, description="Video duration in seconds")
    fps: float = Field(..., gt=0, description="Frames per second")
    resolution: str = Field(..., max_length=50, description="Video resolution (e.g., '1920x1080')")


class Video(VideoBase):
    """Schema for video response."""

    id: int
    upload_date: dt.datetime
    processed: bool
    processing_status: str
    recorded_at: dt.datetime | None = None
    sequence_order: int | None = None
    game_time_offset: float | None = None

    model_config = ConfigDict(from_attributes=True)
