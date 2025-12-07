"""Play model."""

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database import Base

if TYPE_CHECKING:
    from app.models.annotation import Annotation


class PlayType(str, enum.Enum):
    """Type of basketball play."""

    BASKET = "basket"
    MISS = "miss"
    TURNOVER = "turnover"
    REBOUND = "rebound"
    FOUL = "foul"
    SUBSTITUTION = "substitution"
    TIMEOUT = "timeout"


class Play(Base):
    """Detailed play information for an annotation."""

    __tablename__ = "plays"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotations.id", ondelete="CASCADE"), unique=True
    )
    play_type: Mapped[PlayType] = mapped_column(Enum(PlayType))
    player_ids: Mapped[list] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite")
    )  # JSON array of player IDs
    team: Mapped[str] = mapped_column(String(255))
    points_scored: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    annotation: Mapped["Annotation"] = relationship("Annotation", back_populates="play")

    def __repr__(self) -> str:
        return (
            f"<Play(id={self.id}, annotation_id={self.annotation_id}, "
            f"type={self.play_type.value}, team='{self.team}')>"
        )
