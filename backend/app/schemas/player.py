"""Player schemas for API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PlayerBase(BaseModel):
    """Base player schema with common fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Player name")
    jersey_number: int = Field(..., ge=0, le=99, description="Jersey number (0-99)")
    team: str = Field(..., min_length=1, max_length=255, description="Team name")
    notes: str | None = Field(None, description="Optional notes about the player")


class PlayerCreate(PlayerBase):
    """Schema for creating a new player."""

    pass


class PlayerUpdate(BaseModel):
    """Schema for updating a player (all fields optional)."""

    name: str | None = Field(None, min_length=1, max_length=255)
    jersey_number: int | None = Field(None, ge=0, le=99)
    team: str | None = Field(None, min_length=1, max_length=255)
    notes: str | None = None


class Player(PlayerBase):
    """Schema for player response."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PlayerList(BaseModel):
    """Schema for paginated player list response."""

    players: list[Player]
    total: int
    page: int
    page_size: int
    total_pages: int
