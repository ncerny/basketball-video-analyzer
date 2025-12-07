"""Game schemas for API requests and responses."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class GameBase(BaseModel):
    """Base game schema with common fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Game name or identifier")
    date: dt.date = Field(..., description="Game date")
    location: str | None = Field(None, max_length=255, description="Game location or venue")
    home_team: str = Field(..., min_length=1, max_length=255, description="Home team name")
    away_team: str = Field(..., min_length=1, max_length=255, description="Away team name")


class GameCreate(GameBase):
    """Schema for creating a new game."""

    pass


class GameUpdate(BaseModel):
    """Schema for updating a game (all fields optional)."""

    name: str | None = Field(None, min_length=1, max_length=255)
    date: dt.date | None = None
    location: str | None = Field(None, max_length=255)
    home_team: str | None = Field(None, min_length=1, max_length=255)
    away_team: str | None = Field(None, min_length=1, max_length=255)


class Game(GameBase):
    """Schema for game response."""

    id: int
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = ConfigDict(from_attributes=True)


class GameList(BaseModel):
    """Schema for paginated game list response."""

    games: list[Game]
    total: int
    page: int
    page_size: int
    total_pages: int
