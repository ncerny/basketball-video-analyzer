"""Games API endpoints."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.game import Game as GameModel
from app.schemas.game import Game, GameCreate, GameList, GameUpdate

router = APIRouter(prefix="/games", tags=["games"])


@router.post("", response_model=Game, status_code=status.HTTP_201_CREATED)
async def create_game(
    game_data: GameCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Game:
    """Create a new game."""
    game = GameModel(**game_data.model_dump())
    db.add(game)
    await db.commit()
    await db.refresh(game)

    return Game.model_validate(game)


@router.get("", response_model=GameList)
async def list_games(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    team: Annotated[str | None, Query(description="Filter by team (home or away)")] = None,
    date_from: Annotated[date | None, Query(description="Filter games from this date")] = None,
    date_to: Annotated[date | None, Query(description="Filter games until this date")] = None,
    search: Annotated[str | None, Query(description="Search game names")] = None,
) -> GameList:
    """List all games with pagination and optional filtering."""
    # Build query
    stmt = select(GameModel)

    # Apply filters
    if team:
        stmt = stmt.where(
            (GameModel.home_team.ilike(f"%{team}%")) | (GameModel.away_team.ilike(f"%{team}%"))
        )
    if date_from:
        stmt = stmt.where(GameModel.date >= date_from)
    if date_to:
        stmt = stmt.where(GameModel.date <= date_to)
    if search:
        stmt = stmt.where(GameModel.name.ilike(f"%{search}%"))

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination and ordering
    stmt = stmt.order_by(GameModel.date.desc()).offset((page - 1) * page_size).limit(page_size)

    # Execute query
    result = await db.execute(stmt)
    games = result.scalars().all()

    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size

    return GameList(
        games=[Game.model_validate(g) for g in games],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{game_id}", response_model=Game)
async def get_game(
    game_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Game:
    """Get a game by ID."""
    stmt = select(GameModel).where(GameModel.id == game_id)
    result = await db.execute(stmt)
    game = result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {game_id} not found",
        )

    return Game.model_validate(game)


@router.patch("/{game_id}", response_model=Game)
async def update_game(
    game_id: int,
    game_data: GameUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Game:
    """Update a game."""
    stmt = select(GameModel).where(GameModel.id == game_id)
    result = await db.execute(stmt)
    game = result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {game_id} not found",
        )

    # Update only provided fields
    update_data = game_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(game, field, value)

    await db.commit()
    await db.refresh(game)

    return Game.model_validate(game)


@router.delete("/{game_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_game(
    game_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a game and all associated data (videos, annotations, rosters)."""
    stmt = select(GameModel).where(GameModel.id == game_id)
    result = await db.execute(stmt)
    game = result.scalar_one_or_none()

    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with id {game_id} not found",
        )

    await db.delete(game)
    await db.commit()
