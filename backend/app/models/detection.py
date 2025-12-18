"""Player detection model."""

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.player import Player
    from app.models.video import Video


class PlayerDetection(Base):
    """ML detection results for players in video frames."""

    __tablename__ = "player_detections"
    __table_args__ = (
        Index("ix_detection_video_frame", "video_id", "frame_number"),
        Index("ix_detection_tracking", "tracking_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    frame_number: Mapped[int] = mapped_column(Integer)
    player_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.id", ondelete="SET NULL"), nullable=True
    )
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_width: Mapped[float] = mapped_column(Float)
    bbox_height: Mapped[float] = mapped_column(Float)
    tracking_id: Mapped[int] = mapped_column(Integer)
    confidence_score: Mapped[float] = mapped_column(Float)

    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="player_detections")
    player: Mapped["Player | None"] = relationship(
        "Player", back_populates="player_detections"
    )

    def __repr__(self) -> str:
        return (
            f"<PlayerDetection(id={self.id}, video_id={self.video_id}, "
            f"frame={self.frame_number}, tracking_id={self.tracking_id})>"
        )
