"""Annotation-Video junction model."""

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.annotation import Annotation
    from app.models.video import Video


class AnnotationVideo(Base):
    """Junction table mapping annotations to specific video timestamps."""

    __tablename__ = "annotation_videos"
    __table_args__ = (
        UniqueConstraint("annotation_id", "video_id", name="uq_annotation_video"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotations.id", ondelete="CASCADE")
    )
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    video_timestamp_start: Mapped[float] = mapped_column(
        Float
    )  # timestamp in this specific video
    video_timestamp_end: Mapped[float] = mapped_column(
        Float
    )  # timestamp in this specific video

    # Relationships
    annotation: Mapped["Annotation"] = relationship(
        "Annotation", back_populates="annotation_videos"
    )
    video: Mapped["Video"] = relationship("Video", back_populates="annotation_videos")

    def __repr__(self) -> str:
        return (
            f"<AnnotationVideo(annotation_id={self.annotation_id}, video_id={self.video_id}, "
            f"time={self.video_timestamp_start:.1f}-{self.video_timestamp_end:.1f})>"
        )
