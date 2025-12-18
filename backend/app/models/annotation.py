"""Annotation model."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.annotation_video import AnnotationVideo
    from app.models.game import Game
    from app.models.play import Play


class AnnotationType(str, enum.Enum):
    """Type of annotation."""

    PLAY = "play"
    EVENT = "event"
    NOTE = "note"


class CreatedBy(str, enum.Enum):
    """Who created the annotation."""

    AI = "ai"
    USER = "user"


class Annotation(Base):
    """Represents an annotation on the game timeline."""

    __tablename__ = "annotations"
    __table_args__ = (
        Index(
            "ix_annotation_timeline",
            "game_id",
            "game_timestamp_start",
            "game_timestamp_end",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(nullable=True)  # Short title for the annotation
    description: Mapped[str | None] = mapped_column(nullable=True)  # Detailed description/notes
    game_timestamp_start: Mapped[float] = mapped_column(
        Float
    )  # seconds from game start
    game_timestamp_end: Mapped[float] = mapped_column(Float)  # seconds from game start
    annotation_type: Mapped[AnnotationType] = mapped_column(Enum(AnnotationType))
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[CreatedBy] = mapped_column(Enum(CreatedBy))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="annotations")
    annotation_videos: Mapped[list["AnnotationVideo"]] = relationship(
        "AnnotationVideo", back_populates="annotation", cascade="all, delete-orphan"
    )
    play: Mapped["Play | None"] = relationship(
        "Play", back_populates="annotation", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Annotation(id={self.id}, game_id={self.game_id}, "
            f"type={self.annotation_type.value}, "
            f"time={self.game_timestamp_start:.1f}-{self.game_timestamp_end:.1f})>"
        )
