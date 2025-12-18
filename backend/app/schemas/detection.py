"""Detection schemas for API requests and responses."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class DetectionJobRequest(BaseModel):
    """Request to start a detection job."""

    sample_interval: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Extract every Nth frame for detection (higher = faster, less detail)",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of frames to process in each batch",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Minimum confidence score for detections (0.1-1.0)",
    )


class JobProgress(BaseModel):
    """Progress information for a job."""

    current: int = Field(description="Current progress value")
    total: int = Field(description="Total progress value")
    percentage: float = Field(description="Progress as percentage (0-100)")
    message: str = Field(description="Current status message")


class JobResponse(BaseModel):
    """Response with job status information."""

    id: str = Field(description="Unique job ID")
    job_type: str = Field(description="Type of job")
    status: str = Field(description="Job status: pending, processing, completed, failed, cancelled")
    progress: JobProgress
    result: dict | None = Field(None, description="Job result data when completed")
    error: str | None = Field(None, description="Error message if job failed")
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    completed_at: dt.datetime | None = None
    metadata: dict = Field(default_factory=dict)


class DetectionJobResponse(BaseModel):
    """Response when a detection job is started."""

    job_id: str = Field(description="Job ID to use for status polling")
    video_id: int = Field(description="ID of video being processed")
    message: str = Field(description="Status message")


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detection."""

    x: float = Field(description="X coordinate of top-left corner")
    y: float = Field(description="Y coordinate of top-left corner")
    width: float = Field(description="Width of bounding box")
    height: float = Field(description="Height of bounding box")


class Detection(BaseModel):
    """Single player detection result."""

    id: int
    video_id: int
    frame_number: int
    player_id: int | None = Field(None, description="Assigned player ID (null if unassigned)")
    bbox: BoundingBox
    tracking_id: int = Field(description="Tracking ID for this detection")
    confidence_score: float = Field(description="Detection confidence (0-1)")

    model_config = ConfigDict(from_attributes=True)


class VideoDetectionsResponse(BaseModel):
    """Response containing all detections for a video."""

    video_id: int
    total_detections: int
    detections: list[Detection]
    frames_with_detections: int = Field(description="Number of unique frames with detections")


class DetectionStats(BaseModel):
    """Statistics about detections in a video."""

    video_id: int
    total_frames_processed: int
    total_detections: int
    persons_detected: int
    balls_detected: int
    frames_with_detections: int
    avg_detections_per_frame: float
