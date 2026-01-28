"""Video schemas for API requests and responses."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class VideoBase(BaseModel):
    """Base video schema with common fields."""

    game_id: int = Field(..., ge=1, description="ID of the game this video belongs to")
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to video file")
    r2_key: str | None = Field(None, max_length=500, description="Cloudflare R2 storage key")
    thumbnail_path: str | None = Field(None, max_length=500, description="Path to thumbnail image")
    duration_seconds: float = Field(..., gt=0, description="Video duration in seconds")
    fps: float = Field(..., gt=0, description="Frames per second")
    resolution: str = Field(..., max_length=50, description="Video resolution (e.g., '1920x1080')")


class VideoCreate(VideoBase):
    """Schema for creating a new video."""

    pass


class VideoUpdate(BaseModel):
    """Schema for updating a video (all fields optional)."""

    file_path: str | None = Field(None, min_length=1, max_length=500)
    r2_key: str | None = Field(None, max_length=500)
    thumbnail_path: str | None = Field(None, max_length=500)
    duration_seconds: float | None = Field(None, gt=0)
    fps: float | None = Field(None, gt=0)
    resolution: str | None = Field(None, max_length=50)
    processed: bool | None = None
    processing_status: str | None = Field(
        None, description="One of: pending, processing, completed, failed"
    )
    recorded_at: dt.datetime | None = None
    sequence_order: int | None = Field(None, ge=0)
    game_time_offset: float | None = Field(None, ge=0)


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


class VideoList(BaseModel):
    """Schema for paginated video list response."""

    videos: list[Video]
    total: int
    page: int
    page_size: int
    total_pages: int


class JerseyNumberReading(BaseModel):
    id: int
    frame_number: int
    tracking_id: int
    raw_ocr_output: str
    parsed_number: int | None
    confidence: float
    is_valid: bool

    model_config = ConfigDict(from_attributes=True)


class JerseyNumberReadingList(BaseModel):
    video_id: int
    readings: list[JerseyNumberReading]
    total: int


class AggregatedJerseyNumber(BaseModel):
    tracking_id: int
    jersey_number: int | None
    confidence: float
    total_readings: int
    valid_readings: int
    has_conflict: bool
    all_numbers: list[int]


class JerseyNumbersByTrack(BaseModel):
    video_id: int
    tracks: list[AggregatedJerseyNumber]
    total_tracks: int
    tracks_with_numbers: int
