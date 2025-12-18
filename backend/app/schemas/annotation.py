"""Annotation schemas for API requests and responses."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class AnnotationBase(BaseModel):
    """Base annotation schema with common fields."""

    game_id: int = Field(..., ge=1, description="ID of the game this annotation belongs to")
    title: str | None = Field(None, max_length=255, description="Short title for the annotation")
    description: str | None = Field(None, description="Detailed description or notes")
    game_timestamp_start: float = Field(..., ge=0, description="Start time in seconds from game start")
    game_timestamp_end: float = Field(..., ge=0, description="End time in seconds from game start")
    annotation_type: str = Field(..., description="Type of annotation (play, event, note)")
    confidence_score: float | None = Field(None, ge=0, le=1, description="Confidence score for AI annotations (0-1)")
    verified: bool = Field(default=False, description="Whether annotation has been verified by user")
    created_by: str = Field(..., description="Who created the annotation (ai, user)")


class AnnotationCreate(AnnotationBase):
    """Schema for creating a new annotation."""

    video_ids: list[int] = Field(
        default_factory=list,
        description="List of video IDs this annotation spans across"
    )


class AnnotationUpdate(BaseModel):
    """Schema for updating an annotation (all fields optional)."""

    title: str | None = Field(None, max_length=255)
    description: str | None = None
    game_timestamp_start: float | None = Field(None, ge=0)
    game_timestamp_end: float | None = Field(None, ge=0)
    annotation_type: str | None = Field(None, description="play, event, or note")
    confidence_score: float | None = Field(None, ge=0, le=1)
    verified: bool | None = None
    created_by: str | None = Field(None, description="ai or user")
    video_ids: list[int] | None = Field(None, description="Update associated video IDs")


class AnnotationVideoLink(BaseModel):
    """Schema for annotation-video link."""

    video_id: int
    video_timestamp_start: float
    video_timestamp_end: float

    model_config = ConfigDict(from_attributes=True)


class Annotation(AnnotationBase):
    """Schema for annotation response."""

    id: int
    created_at: dt.datetime
    updated_at: dt.datetime
    video_links: list[AnnotationVideoLink] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class AnnotationList(BaseModel):
    """Schema for paginated annotation list response."""

    annotations: list[Annotation]
    total: int
    page: int
    page_size: int
    total_pages: int
