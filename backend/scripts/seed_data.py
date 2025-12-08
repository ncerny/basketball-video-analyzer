#!/usr/bin/env python3
"""
Seed script to populate the database with sample data for testing.

This script creates:
- Sample games with realistic details
- Sample players across multiple teams
- Game rosters linking players to games
- Sample videos with timeline data
- Sample annotations (plays, events, notes)
"""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, text

from app.database import async_session_maker
from app.models import (
    Annotation,
    AnnotationType,
    AnnotationVideo,
    CreatedBy,
    Game,
    GameRoster,
    Player,
    ProcessingStatus,
    TeamSide,
    Video,
)


async def clear_all_data() -> None:
    """Clear all existing data from the database."""
    async with async_session_maker() as session:
        # Delete in reverse dependency order
        await session.execute(text("DELETE FROM annotation_videos"))
        await session.execute(text("DELETE FROM player_detections"))
        await session.execute(text("DELETE FROM plays"))
        await session.execute(text("DELETE FROM annotations"))
        await session.execute(text("DELETE FROM game_rosters"))
        await session.execute(text("DELETE FROM videos"))
        await session.execute(text("DELETE FROM players"))
        await session.execute(text("DELETE FROM games"))
        await session.commit()
        print("✓ Cleared all existing data")


async def create_games() -> list[Game]:
    """Create sample games."""
    games_data = [
        {
            "name": "Warriors vs Lakers - Season Opener",
            "date": date.today() - timedelta(days=7),
            "location": "Chase Center, San Francisco",
            "home_team": "Golden State Warriors",
            "away_team": "Los Angeles Lakers",
        },
        {
            "name": "Celtics vs Nets - Eastern Conference Matchup",
            "date": date.today() - timedelta(days=3),
            "location": "TD Garden, Boston",
            "home_team": "Boston Celtics",
            "away_team": "Brooklyn Nets",
        },
        {
            "name": "Bucks vs Heat - Playoffs Preview",
            "date": date.today() - timedelta(days=1),
            "location": "Fiserv Forum, Milwaukee",
            "home_team": "Milwaukee Bucks",
            "away_team": "Miami Heat",
        },
    ]

    games = []
    async with async_session_maker() as session:
        for game_data in games_data:
            game = Game(**game_data)
            session.add(game)
            games.append(game)

        await session.commit()

        # Refresh to get IDs
        for game in games:
            await session.refresh(game)

    print(f"✓ Created {len(games)} games")
    return games


async def create_players() -> list[Player]:
    """Create sample players."""
    players_data = [
        # Warriors
        {"name": "Stephen Curry", "jersey_number": 30, "team": "Golden State Warriors", "notes": "Point Guard, MVP candidate"},
        {"name": "Klay Thompson", "jersey_number": 11, "team": "Golden State Warriors", "notes": "Shooting Guard"},
        {"name": "Draymond Green", "jersey_number": 23, "team": "Golden State Warriors", "notes": "Forward, Defensive anchor"},
        {"name": "Andrew Wiggins", "jersey_number": 22, "team": "Golden State Warriors", "notes": "Small Forward"},
        {"name": "Kevon Looney", "jersey_number": 5, "team": "Golden State Warriors", "notes": "Center"},
        # Lakers
        {"name": "LeBron James", "jersey_number": 23, "team": "Los Angeles Lakers", "notes": "Forward, All-time great"},
        {"name": "Anthony Davis", "jersey_number": 3, "team": "Los Angeles Lakers", "notes": "Power Forward/Center"},
        {"name": "D'Angelo Russell", "jersey_number": 1, "team": "Los Angeles Lakers", "notes": "Point Guard"},
        {"name": "Austin Reaves", "jersey_number": 15, "team": "Los Angeles Lakers", "notes": "Shooting Guard"},
        # Celtics
        {"name": "Jayson Tatum", "jersey_number": 0, "team": "Boston Celtics", "notes": "Forward, All-Star"},
        {"name": "Jaylen Brown", "jersey_number": 7, "team": "Boston Celtics", "notes": "Guard/Forward"},
        {"name": "Marcus Smart", "jersey_number": 36, "team": "Boston Celtics", "notes": "Point Guard, DPOY"},
        # Nets
        {"name": "Mikal Bridges", "jersey_number": 1, "team": "Brooklyn Nets", "notes": "Forward"},
        {"name": "Cameron Johnson", "jersey_number": 2, "team": "Brooklyn Nets", "notes": "Forward"},
        # Bucks
        {"name": "Giannis Antetokounmpo", "jersey_number": 34, "team": "Milwaukee Bucks", "notes": "Forward, 2x MVP"},
        {"name": "Damian Lillard", "jersey_number": 0, "team": "Milwaukee Bucks", "notes": "Point Guard"},
        # Heat
        {"name": "Jimmy Butler", "jersey_number": 22, "team": "Miami Heat", "notes": "Forward"},
        {"name": "Bam Adebayo", "jersey_number": 13, "team": "Miami Heat", "notes": "Center"},
    ]

    players = []
    async with async_session_maker() as session:
        for player_data in players_data:
            player = Player(**player_data)
            session.add(player)
            players.append(player)

        await session.commit()

        # Refresh to get IDs
        for player in players:
            await session.refresh(player)

    print(f"✓ Created {len(players)} players")
    return players


async def create_game_rosters(games: list[Game], players: list[Player]) -> None:
    """Create game rosters by assigning players to games."""
    async with async_session_maker() as session:
        roster_count = 0

        # Game 1: Warriors vs Lakers
        warriors_players = [p for p in players if p.team == "Golden State Warriors"]
        lakers_players = [p for p in players if p.team == "Los Angeles Lakers"]

        for player in warriors_players:
            roster = GameRoster(
                game_id=games[0].id,
                player_id=player.id,
                team_side=TeamSide.HOME,
            )
            session.add(roster)
            roster_count += 1

        for player in lakers_players:
            roster = GameRoster(
                game_id=games[0].id,
                player_id=player.id,
                team_side=TeamSide.AWAY,
            )
            session.add(roster)
            roster_count += 1

        # Game 2: Celtics vs Nets
        celtics_players = [p for p in players if p.team == "Boston Celtics"]
        nets_players = [p for p in players if p.team == "Brooklyn Nets"]

        for player in celtics_players:
            roster = GameRoster(
                game_id=games[1].id,
                player_id=player.id,
                team_side=TeamSide.HOME,
            )
            session.add(roster)
            roster_count += 1

        for player in nets_players:
            roster = GameRoster(
                game_id=games[1].id,
                player_id=player.id,
                team_side=TeamSide.AWAY,
            )
            session.add(roster)
            roster_count += 1

        # Game 3: Bucks vs Heat
        bucks_players = [p for p in players if p.team == "Milwaukee Bucks"]
        heat_players = [p for p in players if p.team == "Miami Heat"]

        for player in bucks_players:
            roster = GameRoster(
                game_id=games[2].id,
                player_id=player.id,
                team_side=TeamSide.HOME,
            )
            session.add(roster)
            roster_count += 1

        for player in heat_players:
            roster = GameRoster(
                game_id=games[2].id,
                player_id=player.id,
                team_side=TeamSide.AWAY,
            )
            session.add(roster)
            roster_count += 1

        await session.commit()

    print(f"✓ Created {roster_count} game roster entries")


async def create_videos(games: list[Game]) -> list[Video]:
    """Create sample videos for games."""
    videos = []
    async with async_session_maker() as session:
        # Game 1: Warriors vs Lakers - 3 videos
        videos_game1 = [
            Video(
                game_id=games[0].id,
                file_path="/data/videos/warriors_lakers_q1.mp4",
                duration_seconds=720.0,  # 12 minutes
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=1,
                game_time_offset=0.0,
                recorded_at=datetime.now() - timedelta(days=7, hours=3),
            ),
            Video(
                game_id=games[0].id,
                file_path="/data/videos/warriors_lakers_q2.mp4",
                duration_seconds=720.0,
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=2,
                game_time_offset=720.0,
                recorded_at=datetime.now() - timedelta(days=7, hours=2),
            ),
            Video(
                game_id=games[0].id,
                file_path="/data/videos/warriors_lakers_q3.mp4",
                duration_seconds=720.0,
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=3,
                game_time_offset=1440.0,
                recorded_at=datetime.now() - timedelta(days=7, hours=1),
            ),
        ]
        videos.extend(videos_game1)

        # Game 2: Celtics vs Nets - 2 videos
        videos_game2 = [
            Video(
                game_id=games[1].id,
                file_path="/data/videos/celtics_nets_half1.mp4",
                duration_seconds=1440.0,  # 24 minutes (first half)
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=1,
                game_time_offset=0.0,
                recorded_at=datetime.now() - timedelta(days=3, hours=3),
            ),
            Video(
                game_id=games[1].id,
                file_path="/data/videos/celtics_nets_half2.mp4",
                duration_seconds=1440.0,
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=2,
                game_time_offset=1440.0,
                recorded_at=datetime.now() - timedelta(days=3, hours=1),
            ),
        ]
        videos.extend(videos_game2)

        # Game 3: Bucks vs Heat - 1 video (full game)
        videos_game3 = [
            Video(
                game_id=games[2].id,
                file_path="/data/videos/bucks_heat_full.mp4",
                duration_seconds=2880.0,  # 48 minutes
                fps=30.0,
                resolution="1920x1080",
                processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                sequence_order=1,
                game_time_offset=0.0,
                recorded_at=datetime.now() - timedelta(days=1, hours=3),
            ),
        ]
        videos.extend(videos_game3)

        for video in videos:
            session.add(video)

        await session.commit()

        # Refresh to get IDs
        for video in videos:
            await session.refresh(video)

    print(f"✓ Created {len(videos)} videos")
    return videos


async def create_annotations(games: list[Game], videos: list[Video]) -> None:
    """Create sample annotations for games."""
    async with async_session_maker() as session:
        annotations_count = 0

        # Game 1 annotations (Warriors vs Lakers)
        game1_videos = [v for v in videos if v.game_id == games[0].id]

        # Play annotations
        annotations_game1 = [
            # Q1 plays
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=45.0,
                game_timestamp_end=52.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.95,
                verified=True,
                created_by=CreatedBy.AI,
            ),
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=128.5,
                game_timestamp_end=135.2,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.92,
                verified=True,
                created_by=CreatedBy.AI,
            ),
            # Event annotations
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=240.0,
                game_timestamp_end=242.0,
                annotation_type=AnnotationType.EVENT,
                confidence_score=None,
                verified=True,
                created_by=CreatedBy.USER,
            ),
            # Q2 plays
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=785.0,
                game_timestamp_end=792.5,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.88,
                verified=False,
                created_by=CreatedBy.AI,
            ),
            # Note annotation
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=900.0,
                game_timestamp_end=905.0,
                annotation_type=AnnotationType.NOTE,
                confidence_score=None,
                verified=True,
                created_by=CreatedBy.USER,
            ),
            # Q3 plays
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=1520.0,
                game_timestamp_end=1527.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.91,
                verified=False,
                created_by=CreatedBy.AI,
            ),
            Annotation(
                game_id=games[0].id,
                game_timestamp_start=1650.0,
                game_timestamp_end=1658.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.94,
                verified=True,
                created_by=CreatedBy.AI,
            ),
        ]

        for annotation in annotations_game1:
            session.add(annotation)
        await session.flush()

        # Link annotations to videos
        for annotation in annotations_game1:
            # Find which video(s) this annotation spans
            for video in game1_videos:
                video_start = video.game_time_offset or 0.0
                video_end = video_start + video.duration_seconds

                # Check if annotation overlaps with this video
                if (annotation.game_timestamp_start < video_end and
                    annotation.game_timestamp_end > video_start):
                    # Calculate video-specific timestamps
                    video_ts_start = max(0.0, annotation.game_timestamp_start - video_start)
                    video_ts_end = min(
                        video.duration_seconds,
                        annotation.game_timestamp_end - video_start
                    )

                    link = AnnotationVideo(
                        annotation_id=annotation.id,
                        video_id=video.id,
                        video_timestamp_start=video_ts_start,
                        video_timestamp_end=video_ts_end,
                    )
                    session.add(link)

        annotations_count += len(annotations_game1)

        # Game 2 annotations (Celtics vs Nets)
        game2_videos = [v for v in videos if v.game_id == games[1].id]

        annotations_game2 = [
            Annotation(
                game_id=games[1].id,
                game_timestamp_start=120.0,
                game_timestamp_end=127.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.93,
                verified=True,
                created_by=CreatedBy.AI,
            ),
            Annotation(
                game_id=games[1].id,
                game_timestamp_start=680.0,
                game_timestamp_end=685.0,
                annotation_type=AnnotationType.EVENT,
                confidence_score=None,
                verified=True,
                created_by=CreatedBy.USER,
            ),
            Annotation(
                game_id=games[1].id,
                game_timestamp_start=1520.0,
                game_timestamp_end=1528.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.89,
                verified=False,
                created_by=CreatedBy.AI,
            ),
            Annotation(
                game_id=games[1].id,
                game_timestamp_start=2100.0,
                game_timestamp_end=2105.0,
                annotation_type=AnnotationType.NOTE,
                confidence_score=None,
                verified=True,
                created_by=CreatedBy.USER,
            ),
        ]

        for annotation in annotations_game2:
            session.add(annotation)
        await session.flush()

        # Link annotations to videos
        for annotation in annotations_game2:
            for video in game2_videos:
                video_start = video.game_time_offset or 0.0
                video_end = video_start + video.duration_seconds

                if (annotation.game_timestamp_start < video_end and
                    annotation.game_timestamp_end > video_start):
                    # Calculate video-specific timestamps
                    video_ts_start = max(0.0, annotation.game_timestamp_start - video_start)
                    video_ts_end = min(
                        video.duration_seconds,
                        annotation.game_timestamp_end - video_start
                    )

                    link = AnnotationVideo(
                        annotation_id=annotation.id,
                        video_id=video.id,
                        video_timestamp_start=video_ts_start,
                        video_timestamp_end=video_ts_end,
                    )
                    session.add(link)

        annotations_count += len(annotations_game2)

        # Game 3 annotations (Bucks vs Heat)
        game3_videos = [v for v in videos if v.game_id == games[2].id]

        annotations_game3 = [
            Annotation(
                game_id=games[2].id,
                game_timestamp_start=180.0,
                game_timestamp_end=187.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.96,
                verified=True,
                created_by=CreatedBy.AI,
            ),
            Annotation(
                game_id=games[2].id,
                game_timestamp_start=450.0,
                game_timestamp_end=455.0,
                annotation_type=AnnotationType.EVENT,
                confidence_score=None,
                verified=True,
                created_by=CreatedBy.USER,
            ),
            Annotation(
                game_id=games[2].id,
                game_timestamp_start=1200.0,
                game_timestamp_end=1208.0,
                annotation_type=AnnotationType.PLAY,
                confidence_score=0.90,
                verified=False,
                created_by=CreatedBy.AI,
            ),
        ]

        for annotation in annotations_game3:
            session.add(annotation)
        await session.flush()

        # Link annotations to videos
        for annotation in annotations_game3:
            for video in game3_videos:
                video_start = video.game_time_offset or 0.0
                video_end = video_start + video.duration_seconds

                if (annotation.game_timestamp_start < video_end and
                    annotation.game_timestamp_end > video_start):
                    # Calculate video-specific timestamps
                    video_ts_start = max(0.0, annotation.game_timestamp_start - video_start)
                    video_ts_end = min(
                        video.duration_seconds,
                        annotation.game_timestamp_end - video_start
                    )

                    link = AnnotationVideo(
                        annotation_id=annotation.id,
                        video_id=video.id,
                        video_timestamp_start=video_ts_start,
                        video_timestamp_end=video_ts_end,
                    )
                    session.add(link)

        annotations_count += len(annotations_game3)

        await session.commit()

    print(f"✓ Created {annotations_count} annotations with video links")


async def seed_database() -> None:
    """Main function to seed the database with sample data."""
    print("\n" + "="*50)
    print("Basketball Video Analyzer - Database Seeding")
    print("="*50 + "\n")

    print("Clearing existing data...")
    await clear_all_data()
    print()

    print("Creating sample data...")
    games = await create_games()
    players = await create_players()
    await create_game_rosters(games, players)
    videos = await create_videos(games)
    await create_annotations(games, videos)
    print()

    print("="*50)
    print("✓ Database seeding completed successfully!")
    print("="*50)
    print(f"\nCreated:")
    print(f"  - {len(games)} games")
    print(f"  - {len(players)} players")
    print(f"  - {len(videos)} videos")
    print(f"  - Multiple annotations and roster entries")
    print()


def main() -> None:
    """Entry point for the seed script."""
    asyncio.run(seed_database())


if __name__ == "__main__":
    main()
