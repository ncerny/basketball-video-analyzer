import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.video import Video


class BatchStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingBatch(Base):
    __tablename__ = "processing_batches"
    __table_args__ = (
        Index("ix_batch_video_status", "video_id", "detection_status"),
        Index("ix_batch_video_index", "video_id", "batch_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    batch_index: Mapped[int] = mapped_column(Integer)
    frame_start: Mapped[int] = mapped_column(Integer)
    frame_end: Mapped[int] = mapped_column(Integer)

    detection_status: Mapped[BatchStatus] = mapped_column(
        Enum(BatchStatus), default=BatchStatus.PENDING
    )
    ocr_status: Mapped[BatchStatus] = mapped_column(Enum(BatchStatus), default=BatchStatus.PENDING)

    detection_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ocr_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    video: Mapped["Video"] = relationship("Video", back_populates="processing_batches")

    def __repr__(self) -> str:
        return (
            f"<ProcessingBatch(id={self.id}, video_id={self.video_id}, "
            f"batch={self.batch_index}, frames={self.frame_start}-{self.frame_end}, "
            f"detection={self.detection_status.value}, ocr={self.ocr_status.value})>"
        )
