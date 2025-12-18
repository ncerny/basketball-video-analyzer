"""Game Rosters API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.game import Game as GameModel
from app.models.game_roster import GameRoster as GameRosterModel, TeamSide
from app.models.player import Player as PlayerModel
from app.schemas.game_roster import (
    GameRoster,
    GameRosterCreate,
    GameRosterList,
    GameRosterUpdate,
    GameRosterWithDetails,
)

router = APIRouter(prefix="/game-rosters", tags=["game-rosters"])


@router.post("", response_model=GameRoster, status_code=status.HTTP_201_CREATED)
async def add_player_to_game(
    roster_data: GameRosterCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GameRoster:
    """Add a player to a game roster."""
    # Verify game exists
    game_stmt = select(GameModel).where(GameModel.id == roster_data.game_id)
    game_result = await db.execute(game_stmt)
    game = game_result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {roster_data.game_id} not found",
        )

    # Verify player exists
    player_stmt = select(PlayerModel).where(PlayerModel.id == roster_data.player_id)
    player_result = await db.execute(player_stmt)
    player = player_result.scalar_one_or_none()

    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with id {roster_data.player_id} not found",
        )

    # Check if player is already on this game's roster
    existing_stmt = select(GameRosterModel).where(
        GameRosterModel.game_id == roster_data.game_id,
        GameRosterModel.player_id == roster_data.player_id,
    )
    existing_result = await db.execute(existing_stmt)
    existing_roster = existing_result.scalar_one_or_none()

    if existing_roster:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Player {roster_data.player_id} is already on the roster for game {roster_data.game_id}",
        )

    # Validate team_side
    try:
        team_side_enum = TeamSide(roster_data.team_side.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid team_side: {roster_data.team_side}. Must be one of: home, away",
        )

    # Create roster entry
    roster = GameRosterModel(
        game_id=roster_data.game_id,
        player_id=roster_data.player_id,
        team_side=team_side_enum,
        jersey_number_override=roster_data.jersey_number_override,
    )
    db.add(roster)
    await db.commit()
    await db.refresh(roster)

    return GameRoster.model_validate(roster)


@router.get("", response_model=GameRosterList)
async def list_game_rosters(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    game_id: Annotated[int | None, Query(description="Filter by game ID")] = None,
    player_id: Annotated[int | None, Query(description="Filter by player ID")] = None,
    team_side: Annotated[str | None, Query(description="Filter by team side (home/away)")] = None,
) -> GameRosterList:
    """List all game roster entries with pagination and optional filtering."""
    # Build query
    stmt = select(GameRosterModel)

    # Apply filters
    if game_id is not None:
        stmt = stmt.where(GameRosterModel.game_id == game_id)
    if player_id is not None:
        stmt = stmt.where(GameRosterModel.player_id == player_id)
    if team_side is not None:
        try:
            team_side_enum = TeamSide(team_side.lower())
            stmt = stmt.where(GameRosterModel.team_side == team_side_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid team_side: {team_side}. Must be one of: home, away",
            )

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination and ordering
    stmt = stmt.order_by(GameRosterModel.game_id, GameRosterModel.id).offset((page - 1) * page_size).limit(page_size)

    # Execute query
    result = await db.execute(stmt)
    rosters = result.scalars().all()

    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size

    return GameRosterList(
        rosters=[GameRoster.model_validate(r) for r in rosters],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/with-details", response_model=list[GameRosterWithDetails])
async def list_game_rosters_with_details(
    db: Annotated[AsyncSession, Depends(get_db)],
    game_id: Annotated[int, Query(description="Game ID to get roster for")],
) -> list[GameRosterWithDetails]:
    """Get game roster with player details for a specific game."""
    # Verify game exists
    game_stmt = select(GameModel).where(GameModel.id == game_id)
    game_result = await db.execute(game_stmt)
    game = game_result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {game_id} not found",
        )

    # Get roster with player details using join
    stmt = (
        select(
            GameRosterModel.id,
            GameRosterModel.game_id,
            GameRosterModel.player_id,
            GameRosterModel.team_side,
            GameRosterModel.jersey_number_override,
            PlayerModel.name.label("player_name"),
            PlayerModel.jersey_number.label("player_default_jersey"),
            PlayerModel.team.label("player_team"),
        )
        .join(PlayerModel, GameRosterModel.player_id == PlayerModel.id)
        .where(GameRosterModel.game_id == game_id)
        .order_by(GameRosterModel.team_side, PlayerModel.jersey_number)
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        GameRosterWithDetails(
            id=row.id,
            game_id=row.game_id,
            player_id=row.player_id,
            team_side=row.team_side.value,
            jersey_number_override=row.jersey_number_override,
            player_name=row.player_name,
            player_default_jersey=row.player_default_jersey,
            player_team=row.player_team,
        )
        for row in rows
    ]


@router.get("/{roster_id}", response_model=GameRoster)
async def get_game_roster(
    roster_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GameRoster:
    """Get a game roster entry by ID."""
    stmt = select(GameRosterModel).where(GameRosterModel.id == roster_id)
    result = await db.execute(stmt)
    roster = result.scalar_one_or_none()

    if not roster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game roster entry with id {roster_id} not found",
        )

    return GameRoster.model_validate(roster)


@router.patch("/{roster_id}", response_model=GameRoster)
async def update_game_roster(
    roster_id: int,
    roster_data: GameRosterUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GameRoster:
    """Update a game roster entry (team side or jersey number override)."""
    stmt = select(GameRosterModel).where(GameRosterModel.id == roster_id)
    result = await db.execute(stmt)
    roster = result.scalar_one_or_none()

    if not roster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game roster entry with id {roster_id} not found",
        )

    # Update only provided fields
    update_data = roster_data.model_dump(exclude_unset=True)

    # Convert team_side string to enum if provided
    if "team_side" in update_data and update_data["team_side"] is not None:
        try:
            update_data["team_side"] = TeamSide(update_data["team_side"].lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid team_side: {update_data['team_side']}. Must be one of: home, away",
            )

    for field, value in update_data.items():
        setattr(roster, field, value)

    await db.commit()
    await db.refresh(roster)

    return GameRoster.model_validate(roster)


@router.delete("/{roster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_player_from_game(
    roster_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove a player from a game roster."""
    stmt = select(GameRosterModel).where(GameRosterModel.id == roster_id)
    result = await db.execute(stmt)
    roster = result.scalar_one_or_none()

    if not roster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game roster entry with id {roster_id} not found",
        )

    await db.delete(roster)
    await db.commit()
