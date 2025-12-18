"""Player model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.detection import PlayerDetection
    from app.models.game_roster import GameRoster


class Player(Base):
    """Represents a basketball player."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    jersey_number: Mapped[int] = mapped_column(Integer)
    team: Mapped[str] = mapped_column(String(255))  # team name/identifier
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    # Relationships
    game_rosters: Mapped[list["GameRoster"]] = relationship(
        "GameRoster", back_populates="player", cascade="all, delete-orphan"
    )
    player_detections: Mapped[list["PlayerDetection"]] = relationship(
        "PlayerDetection", back_populates="player"
    )

    def __repr__(self) -> str:
        return f"<Player(id={self.id}, name='{self.name}', number={self.jersey_number}, team='{self.team}')>"
