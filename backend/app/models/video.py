"""Video model."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.annotation_video import AnnotationVideo
    from app.models.detection import PlayerDetection
    from app.models.game import Game


class ProcessingStatus(str, enum.Enum):
    """Video processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    """Represents a video file for a game."""

    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_video_timeline", "game_id", "game_time_offset"),
        Index("ix_video_game_id", "game_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    file_path: Mapped[str] = mapped_column(String(500))
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float)
    fps: Mapped[float] = mapped_column(Float)
    resolution: Mapped[str] = mapped_column(String(50))  # e.g., "1920x1080"
    upload_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    processed: Mapped[bool] = mapped_column(default=False)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING
    )

    # Timeline synchronization fields (nullable until determined)
    recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sequence_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_time_offset: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # seconds from game start

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="videos")
    annotation_videos: Mapped[list["AnnotationVideo"]] = relationship(
        "AnnotationVideo", back_populates="video", cascade="all, delete-orphan"
    )
    player_detections: Mapped[list["PlayerDetection"]] = relationship(
        "PlayerDetection", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Video(id={self.id}, game_id={self.game_id}, "
            f"sequence_order={self.sequence_order}, duration={self.duration_seconds:.1f}s)>"
        )
