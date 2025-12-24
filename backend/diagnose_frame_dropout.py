import asyncio
from pathlib import Path
from app.ml.yolo_detector import YOLODetector
from app.services.frame_extractor import FrameExtractor
import sys


async def main():
    video_path = Path("videos/game_41/game41_20251214_180538_IMG_1517.MOV")

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print(f"✓ Found video: {video_path}")
    print()

    detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.35, device="cpu")

    extractor = FrameExtractor(str(video_path))
    metadata = extractor.get_metadata()
    print(
        f"Video: {metadata.width}x{metadata.height} @ {metadata.fps:.2f}fps, {metadata.total_frames} frames"
    )
    print()
    print("Running raw YOLO detection on frames 0-10...")
    print()

    frames_to_check = list(range(11))

    for frame_num in frames_to_check:
        extracted = extractor.extract_frame(frame_num)
        if not extracted:
            print(f"❌ Failed to extract frame {frame_num}")
            continue

        detections = detector.detect(extracted.frame, frame_num)

        persons = [d for d in detections.detections if d.class_id == 0]
        balls = [d for d in detections.detections if d.class_id == 32]

        normal_persons = [p for p in persons if p.bbox.width * p.bbox.height < 100000]
        large_objs = [p for p in persons if p.bbox.width * p.bbox.height >= 100000]

        print(
            f"Frame {frame_num:2d}: {len(persons):2d} persons ({len(normal_persons):2d} normal, {len(large_objs):2d} large), {len(balls):2d} balls"
        )

        if len(normal_persons) < 5:
            print(f"  ⚠️  Only {len(normal_persons)} normal-sized persons detected!")
            for i, p in enumerate(
                sorted(normal_persons, key=lambda d: d.confidence, reverse=True), 1
            ):
                area = p.bbox.width * p.bbox.height
                print(
                    f"    #{i}: {p.bbox.width:.0f}x{p.bbox.height:.0f} (area={area:.0f}) conf={p.confidence:.3f}"
                )

    extractor.close()
    print()
    print("=" * 60)
    print("CONCLUSION:")
    print("If YOLO detects 10+ normal persons in frame 0 but <5 in frame 1,")
    print("the problem is YOLO itself (not tracking/filtering).")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
