#!/usr/bin/env python3
"""Integration test for SAM3 video tracking.

Usage:
    python -m scripts.test_sam3_tracking <video_path>

This script tests SAM3 tracking on a real video and reports:
- Number of unique track IDs (should be ~10 for basketball, not 50+)
- Track ID frequency distribution
- Processing time
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAM3 tracking on video")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument(
        "--sample-interval", type=int, default=3, help="Sample every Nth frame"
    )
    parser.add_argument(
        "--prompt", type=str, default="basketball player", help="Text prompt"
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

    config = SAM3TrackerConfig(prompt=args.prompt)
    tracker = SAM3VideoTracker(config)

    logger.info(f"Processing video: {args.video_path}")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Sample interval: {args.sample_interval}")
    logger.info("-" * 50)

    start_time = time.time()
    all_track_ids = []
    frame_count = 0

    for frame_detections in tracker.process_video(
        args.video_path, sample_interval=args.sample_interval
    ):
        frame_count += 1
        frame_track_ids = [d.tracking_id for d in frame_detections.detections]
        all_track_ids.extend(frame_track_ids)

        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames...")

    elapsed = time.time() - start_time

    # Analysis
    unique_ids = set(all_track_ids)
    id_counts = Counter(all_track_ids)

    logger.info("-" * 50)
    logger.info("RESULTS")
    logger.info("-" * 50)
    logger.info(f"Frames processed: {frame_count}")
    if frame_count > 0:
        logger.info(f"Processing time: {elapsed:.1f}s ({elapsed/frame_count:.2f}s/frame)")
    else:
        logger.info(f"Processing time: {elapsed:.1f}s (no frames processed)")
        logger.warning("No frames were processed - check video file")
        sys.exit(1)
    logger.info(f"Unique track IDs: {len(unique_ids)}")
    logger.info(f"Total detections: {len(all_track_ids)}")

    logger.info("\nTrack ID frequency (top 15):")
    for track_id, count in id_counts.most_common(15):
        logger.info(f"  Track {track_id}: {count} detections")

    # Success criteria
    if len(unique_ids) <= 15:
        logger.info("\nPASS: Reasonable number of track IDs")
    else:
        logger.warning(f"\nWARN: {len(unique_ids)} track IDs (expected ~10)")


if __name__ == "__main__":
    main()
