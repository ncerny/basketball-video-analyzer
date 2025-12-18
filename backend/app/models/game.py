"""Game model."""

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.annotation import Annotation
    from app.models.game_roster import GameRoster
    from app.models.video import Video


class Game(Base):
    """Represents a basketball game."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    date: Mapped[date] = mapped_column(Date)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    home_team: Mapped[str] = mapped_column(String(255))
    away_team: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="game", cascade="all, delete-orphan"
    )
    game_rosters: Mapped[list["GameRoster"]] = relationship(
        "GameRoster", back_populates="game", cascade="all, delete-orphan"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="game", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Game(id={self.id}, name='{self.name}', date={self.date})>"
