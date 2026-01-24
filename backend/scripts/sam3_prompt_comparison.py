#!/usr/bin/env python3
"""Compare SAM3 tracking with different prompts and thresholds.

Usage:
    uv run python -m scripts.sam3_prompt_comparison <video_path>
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
]


def get_color(track_id: int) -> tuple:
    return COLORS[track_id % len(COLORS)]


def draw_detections(frame: np.ndarray, detections: list, frame_number: int,
                    prompt: str, threshold: float) -> np.ndarray:
    """Draw bounding boxes and tracking IDs on frame."""
    annotated = frame.copy()

    for det in detections:
        color = get_color(det.tracking_id)

        x1, y1 = int(det.bbox.x), int(det.bbox.y)
        x2, y2 = int(det.bbox.x + det.bbox.width), int(det.bbox.y + det.bbox.height)

        # Calculate box area for filtering display
        box_area = det.bbox.width * det.bbox.height
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = box_area / frame_area

        # Draw box (thicker if suspiciously large)
        thickness = 4 if area_ratio > 0.15 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw tracking ID label
        label = f"ID:{det.tracking_id} ({det.confidence:.2f})"
        if area_ratio > 0.15:
            label += " [LARGE]"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw config info at top
    info1 = f"Prompt: '{prompt}' | Threshold: {threshold}"
    info2 = f"Frame: {frame_number} | Detections: {len(detections)}"
    cv2.putText(annotated, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated


def run_single_config(video_path: Path, prompt: str, threshold: float,
                      output_dir: Path, sample_interval: int, max_frames: int):
    """Run SAM3 with a single configuration."""
    from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

    config_name = f"{prompt.replace(' ', '_')}_t{threshold}"
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: prompt='{prompt}', threshold={threshold}")
    logger.info(f"Output: {config_dir}")
    logger.info(f"{'='*60}")

    config = SAM3TrackerConfig(prompt=prompt, confidence_threshold=threshold)
    tracker = SAM3VideoTracker(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video")
        return

    frame_count = 0
    all_track_ids = set()
    large_box_count = 0

    try:
        for frame_detections in tracker.process_video(video_path, sample_interval=sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_detections.frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            # Count large boxes and track IDs
            frame_area = frame.shape[0] * frame.shape[1]
            for det in frame_detections.detections:
                all_track_ids.add(det.tracking_id)
                box_area = det.bbox.width * det.bbox.height
                if box_area / frame_area > 0.15:
                    large_box_count += 1

            annotated = draw_detections(
                frame, frame_detections.detections,
                frame_detections.frame_number, prompt, threshold
            )

            output_path = config_dir / f"frame_{frame_detections.frame_number:06d}.jpg"
            cv2.imwrite(str(output_path), annotated)

            logger.info(f"  Frame {frame_detections.frame_number}: {len(frame_detections.detections)} detections")

            frame_count += 1
            if frame_count >= max_frames:
                break
    finally:
        cap.release()

    # Summary
    logger.info(f"\nSummary for '{prompt}' (threshold={threshold}):")
    logger.info(f"  Frames processed: {frame_count}")
    logger.info(f"  Unique track IDs: {len(all_track_ids)}")
    logger.info(f"  Large boxes (>15% frame): {large_box_count}")
    logger.info(f"  Track IDs: {sorted(all_track_ids)}")

    return {
        "prompt": prompt,
        "threshold": threshold,
        "frames": frame_count,
        "unique_ids": len(all_track_ids),
        "large_boxes": large_box_count,
        "output_dir": str(config_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SAM3 prompts and thresholds")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument("--output-dir", type=Path, default=Path("sam3_comparison"),
                       help="Output directory")
    parser.add_argument("--sample-interval", type=int, default=2000,
                       help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=5,
                       help="Max frames per config")
    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test configurations
    configs = [
        # Original
        ("basketball player", 0.25),
        ("basketball player", 0.5),

        # More specific prompts
        ("person", 0.4),
        ("individual person standing", 0.4),
        ("single person in sports jersey", 0.4),
        ("one basketball player", 0.5),
    ]

    results = []
    for prompt, threshold in configs:
        try:
            result = run_single_config(
                args.video_path, prompt, threshold,
                args.output_dir, args.sample_interval, args.max_frames
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed for prompt='{prompt}': {e}")

    # Final comparison
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Prompt':<35} {'Thresh':>6} {'IDs':>5} {'Large':>6}")
    logger.info("-" * 60)
    for r in results:
        logger.info(f"{r['prompt']:<35} {r['threshold']:>6.2f} {r['unique_ids']:>5} {r['large_boxes']:>6}")

    logger.info(f"\nView results: open {args.output_dir}")


if __name__ == "__main__":
    main()
