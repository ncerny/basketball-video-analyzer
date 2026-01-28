"""Processing job model for database-backed job queue.

This model enables a separate worker process to pick up and execute
ML processing jobs (like video detection) independently from the API server.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.video import Video


class JobStatus(str, enum.Enum):
    """Status of a processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Types of processing jobs."""

    VIDEO_DETECTION = "video_detection"
    TRACK_REPROCESS = "track_reprocess"


class ProcessingJob(Base):
    """Represents a background processing job in the database.

    This replaces the in-memory JobManager for ML workloads,
    allowing a separate worker process to poll for and execute jobs.
    """

    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("ix_job_status_created", "status", "created_at"),
        Index("ix_job_video_id", "video_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    job_type: Mapped[JobType] = mapped_column(Enum(JobType))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING)

    # Foreign key to video (nullable for non-video jobs)
    video_id: Mapped[int | None] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=True
    )

    # Job parameters (stored as JSON)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Progress tracking
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[str] = mapped_column(String(500), default="")

    # Result/error storage
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Worker tracking (which worker picked up the job)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationship
    video: Mapped["Video | None"] = relationship("Video")

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.progress_total == 0:
            return 0.0
        return (self.progress_current / self.progress_total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "video_id": self.video_id,
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
                "percentage": self.progress_percentage,
                "message": self.progress_message,
            },
            "result": self.result,
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "worker_id": self.worker_id,
        }

    def __repr__(self) -> str:
        return (
            f"<ProcessingJob(id={self.id[:8]}..., type={self.job_type.value}, "
            f"status={self.status.value}, video_id={self.video_id})>"
        )
