"""Players API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.player import Player as PlayerModel
from app.schemas.player import Player, PlayerCreate, PlayerList, PlayerUpdate

router = APIRouter(prefix="/players", tags=["players"])


@router.post("", response_model=Player, status_code=status.HTTP_201_CREATED)
async def create_player(
    player_data: PlayerCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Player:
    """Create a new player."""
    # Check for duplicate player (same name, jersey number, and team)
    stmt = select(PlayerModel).where(
        PlayerModel.name == player_data.name,
        PlayerModel.jersey_number == player_data.jersey_number,
        PlayerModel.team == player_data.team,
    )
    result = await db.execute(stmt)
    existing_player = result.scalar_one_or_none()

    if existing_player:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Player '{player_data.name}' with jersey number {player_data.jersey_number} on team '{player_data.team}' already exists",
        )

    player = PlayerModel(**player_data.model_dump())
    db.add(player)
    await db.commit()
    await db.refresh(player)

    return Player.model_validate(player)


@router.get("", response_model=PlayerList)
async def list_players(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    team: Annotated[str | None, Query(description="Filter by team")] = None,
    search: Annotated[str | None, Query(description="Search player names")] = None,
) -> PlayerList:
    """List all players with pagination and optional filtering."""
    # Build query
    stmt = select(PlayerModel)

    # Apply filters
    if team:
        stmt = stmt.where(PlayerModel.team == team)
    if search:
        stmt = stmt.where(PlayerModel.name.ilike(f"%{search}%"))

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination
    stmt = stmt.order_by(PlayerModel.name).offset((page - 1) * page_size).limit(page_size)

    # Execute query
    result = await db.execute(stmt)
    players = result.scalars().all()

    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size

    return PlayerList(
        players=[Player.model_validate(p) for p in players],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{player_id}", response_model=Player)
async def get_player(
    player_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Player:
    """Get a player by ID."""
    stmt = select(PlayerModel).where(PlayerModel.id == player_id)
    result = await db.execute(stmt)
    player = result.scalar_one_or_none()

    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with id {player_id} not found",
        )

    return Player.model_validate(player)


@router.patch("/{player_id}", response_model=Player)
async def update_player(
    player_id: int,
    player_data: PlayerUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Player:
    """Update a player."""
    stmt = select(PlayerModel).where(PlayerModel.id == player_id)
    result = await db.execute(stmt)
    player = result.scalar_one_or_none()

    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with id {player_id} not found",
        )

    # Update only provided fields
    update_data = player_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(player, field, value)

    await db.commit()
    await db.refresh(player)

    return Player.model_validate(player)


@router.delete("/{player_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_player(
    player_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a player."""
    stmt = select(PlayerModel).where(PlayerModel.id == player_id)
    result = await db.execute(stmt)
    player = result.scalar_one_or_none()

    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with id {player_id} not found",
        )

    await db.delete(player)
    await db.commit()
