"""Game roster model."""

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.game import Game
    from app.models.player import Player


class TeamSide(str, enum.Enum):
    """Team side for game roster."""

    HOME = "home"
    AWAY = "away"


class GameRoster(Base):
    """Junction table for game-player assignments with jersey overrides."""

    __tablename__ = "game_rosters"
    __table_args__ = (UniqueConstraint("game_id", "player_id", name="uq_game_player"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"))
    team_side: Mapped[TeamSide] = mapped_column(Enum(TeamSide))
    jersey_number_override: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="game_rosters")
    player: Mapped["Player"] = relationship("Player", back_populates="game_rosters")

    def __repr__(self) -> str:
        return (
            f"<GameRoster(game_id={self.game_id}, player_id={self.player_id}, "
            f"team_side={self.team_side.value})>"
        )
