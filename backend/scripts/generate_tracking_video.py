#!/usr/bin/env python3
"""Generate a video with SAM3 tracking visualization.

Usage:
    uv run python -m scripts.generate_tracking_video <video_path> [options]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Color palette for different tracking IDs (BGR format)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (255, 255, 128),
]


def get_color(track_id: int) -> tuple:
    return COLORS[track_id % len(COLORS)]


def draw_detections(frame: np.ndarray, detections: list, frame_number: int) -> np.ndarray:
    """Draw bounding boxes and tracking IDs on frame."""
    annotated = frame.copy()

    for det in detections:
        color = get_color(det.tracking_id)

        x1, y1 = int(det.bbox.x), int(det.bbox.y)
        x2, y2 = int(det.bbox.x + det.bbox.width), int(det.bbox.y + det.bbox.height)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw tracking ID label
        label = f"ID:{det.tracking_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8),
                     (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw frame info
    info = f"Frame: {frame_number} | Players: {len(detections)}"
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated


def generate_tracking_video(
    video_path: Path,
    output_path: Path,
    prompt: str,
    threshold: float,
    max_frames: int,
    start_frame: int,
):
    """Generate video with tracking visualization."""
    from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

    logger.info(f"Input: {video_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Prompt: '{prompt}', Threshold: {threshold}")
    logger.info(f"Frames: {start_frame} to {start_frame + max_frames}")

    config = SAM3TrackerConfig(prompt=prompt, confidence_threshold=threshold)
    tracker = SAM3VideoTracker(config)

    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    all_track_ids = set()

    try:
        logger.info("Processing video with SAM3...")
        for frame_detections in tracker.process_video(
            video_path,
            sample_interval=1,
            max_frames=max_frames,
            start_frame=start_frame,
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_detections.frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            # Track unique IDs
            for det in frame_detections.detections:
                all_track_ids.add(det.tracking_id)

            # Draw detections
            annotated = draw_detections(
                frame, frame_detections.detections, frame_detections.frame_number
            )
            out.write(annotated)

            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    finally:
        cap.release()
        out.release()

    logger.info(f"\n=== VIDEO GENERATED ===")
    logger.info(f"Output: {output_path}")
    logger.info(f"Frames: {frame_count}")
    logger.info(f"Unique track IDs: {len(all_track_ids)}")
    logger.info(f"Track IDs: {sorted(all_track_ids)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate tracking visualization video")
    parser.add_argument("video_path", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--prompt", type=str, default="basketball player")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    output_path = args.output or Path(f"tracked_sam3_{args.video_path.stem}.mp4")

    generate_tracking_video(
        args.video_path,
        output_path,
        args.prompt,
        args.threshold,
        args.max_frames,
        args.start_frame,
    )


if __name__ == "__main__":
    main()
