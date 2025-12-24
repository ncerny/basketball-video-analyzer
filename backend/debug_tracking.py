"""Debug script to understand tracking ID duplicates."""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

async def check_duplicates():
    engine = create_async_engine("sqlite+aiosqlite:///./basketball_analyzer.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Find a frame with duplicates
        query = text("""
            SELECT frame_number, tracking_id, COUNT(*) as count
            FROM player_detections
            WHERE video_id = 7
            GROUP BY frame_number, tracking_id
            HAVING count > 1
            LIMIT 1
        """)
        result = await session.execute(query)
        dup = result.fetchone()
        
        if dup:
            frame, track_id, count = dup
            print(f"\nFrame {frame} has {count} detections with tracking_id={track_id}")
            
            # Get details
            detail_query = text("""
                SELECT id, bbox_x, bbox_y, bbox_width, bbox_height, confidence_score
                FROM player_detections
                WHERE video_id = 7 AND frame_number = :frame AND tracking_id = :track_id
                ORDER BY id
            """)
            details = await session.execute(detail_query, {"frame": frame, "track_id": track_id})
            
            print("\nDetections:")
            for det in details.fetchall():
                print(f"  ID {det[0]}: pos=({det[1]:.0f}, {det[2]:.0f}) size={det[3]:.0f}x{det[4]:.0f} conf={det[5]:.3f}")

asyncio.run(check_duplicates())
