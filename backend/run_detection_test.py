import asyncio
from pathlib import Path
from app.services.detection_pipeline import DetectionPipeline, DetectionPipelineConfig
from app.database import SessionLocal


async def main():
    config = DetectionPipelineConfig(
        sample_interval=3,
        batch_size=8,
        confidence_threshold=0.35,
        track_activation_threshold=0.25,
        device="cpu",
        delete_existing=True,
        enable_tracking=True,
        tracking_buffer_seconds=5.0,
        tracking_iou_threshold=0.35,
        enable_court_detection=False,
        max_seconds=60,
        enable_track_merging=True,
    )

    video_storage_path = Path("videos")

    db = SessionLocal()

    try:
        pipeline = DetectionPipeline(
            db=db,
            config=config,
            video_storage_path=video_storage_path,
        )

        print("Starting detection on video 7 (max 60 seconds)...")
        print(
            f"Config: confidence={config.confidence_threshold}, track_activation={config.track_activation_threshold}"
        )
        print(f"Court detection: {config.enable_court_detection}")
        print()

        result = await pipeline.process_video(video_id=7)

        print()
        print("=" * 60)
        print("DETECTION COMPLETE")
        print("=" * 60)
        print(f"Frames processed: {result.total_frames_processed}")
        print(f"Total detections: {result.total_detections}")
        print(f"Persons detected: {result.persons_detected}")
        print(f"Balls detected: {result.balls_detected}")
        if result.error:
            print(f"Error: {result.error}")
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
