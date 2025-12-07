"""Game roster schemas for API requests and responses."""

from pydantic import BaseModel, ConfigDict, Field


class GameRosterBase(BaseModel):
    """Base game roster schema with common fields."""

    game_id: int = Field(..., ge=1, description="ID of the game")
    player_id: int = Field(..., ge=1, description="ID of the player")
    team_side: str = Field(..., description="Team side (home or away)")
    jersey_number_override: int | None = Field(
        None, ge=0, le=99, description="Override player's default jersey number for this game"
    )


class GameRosterCreate(GameRosterBase):
    """Schema for adding a player to a game roster."""

    pass


class GameRosterUpdate(BaseModel):
    """Schema for updating a game roster entry (all fields optional)."""

    team_side: str | None = Field(None, description="Team side (home or away)")
    jersey_number_override: int | None = Field(None, ge=0, le=99)


class GameRoster(GameRosterBase):
    """Schema for game roster response."""

    id: int

    model_config = ConfigDict(from_attributes=True)


class GameRosterWithDetails(BaseModel):
    """Schema for game roster with player details."""

    id: int
    game_id: int
    player_id: int
    team_side: str
    jersey_number_override: int | None
    player_name: str
    player_default_jersey: int
    player_team: str

    model_config = ConfigDict(from_attributes=True)


class GameRosterList(BaseModel):
    """Schema for paginated game roster list response."""

    rosters: list[GameRoster]
    total: int
    page: int
    page_size: int
    total_pages: int
