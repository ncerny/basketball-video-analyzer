from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.video import Video


class JerseyNumber(Base):
    __tablename__ = "jersey_numbers"
    __table_args__ = (
        Index("ix_jersey_video_frame", "video_id", "frame_number"),
        Index("ix_jersey_tracking", "tracking_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    frame_number: Mapped[int] = mapped_column(Integer)
    tracking_id: Mapped[int] = mapped_column(Integer)

    raw_ocr_output: Mapped[str] = mapped_column(String(255))
    parsed_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    is_valid: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="jersey_numbers")

    def __repr__(self) -> str:
        return (
            f"<JerseyNumber(id={self.id}, tracking_id={self.tracking_id}, "
            f"parsed={self.parsed_number}, valid={self.is_valid})>"
        )
