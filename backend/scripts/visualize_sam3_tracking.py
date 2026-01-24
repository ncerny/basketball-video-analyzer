#!/usr/bin/env python3
"""Visualize SAM3 tracking results with bounding boxes on frames.

Usage:
    uv run python -m scripts.visualize_sam3_tracking <video_path> [--output-dir <dir>] [--sample-interval <n>]

Example:
    uv run python -m scripts.visualize_sam3_tracking videos/game_41/game41_20251214_180538_IMG_1517.MOV --sample-interval 500
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
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light blue
    (128, 255, 0),  # Lime
    (255, 0, 128),  # Pink
    (0, 255, 128),  # Teal
]


def get_color(track_id: int) -> tuple:
    """Get consistent color for a tracking ID."""
    return COLORS[track_id % len(COLORS)]


def draw_detections(frame: np.ndarray, detections: list, frame_number: int) -> np.ndarray:
    """Draw bounding boxes and tracking IDs on frame."""
    annotated = frame.copy()
    
    for det in detections:
        color = get_color(det.tracking_id)
        
        # Draw bounding box
        x1, y1 = int(det.bbox.x), int(det.bbox.y)
        x2, y2 = int(det.bbox.x + det.bbox.width), int(det.bbox.y + det.bbox.height)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw tracking ID label
        label = f"ID:{det.tracking_id} ({det.confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for label
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw frame info
    info = f"Frame: {frame_number} | Detections: {len(detections)}"
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SAM3 tracking results")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument("--output-dir", type=Path, default=Path("sam3_viz"), 
                       help="Output directory for annotated frames")
    parser.add_argument("--sample-interval", type=int, default=500,
                       help="Sample every Nth frame (default: 500 for speed)")
    parser.add_argument("--max-frames", type=int, default=20,
                       help="Maximum frames to process (default: 20)")
    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig
    from app.ml.sam3_frame_extractor import SAM3FrameExtractor

    logger.info(f"Processing video: {args.video_path}")
    logger.info(f"Sample interval: {args.sample_interval}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("-" * 50)

    # Initialize tracker
    config = SAM3TrackerConfig(prompt="basketball player", confidence_threshold=0.25)
    tracker = SAM3VideoTracker(config)

    # Open video to read frames for visualization
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video_path}")
        sys.exit(1)

    frame_count = 0
    saved_frames = []

    try:
        for frame_detections in tracker.process_video(args.video_path, sample_interval=args.sample_interval):
            # Read the original frame at this position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_detections.frame_number)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame {frame_detections.frame_number}")
                continue
            
            # Draw detections on frame
            annotated = draw_detections(frame, frame_detections.detections, frame_detections.frame_number)
            
            # Save annotated frame
            output_path = args.output_dir / f"frame_{frame_detections.frame_number:06d}.jpg"
            cv2.imwrite(str(output_path), annotated)
            saved_frames.append(output_path)
            
            logger.info(f"Saved: {output_path.name} ({len(frame_detections.detections)} detections)")
            
            frame_count += 1
            if frame_count >= args.max_frames:
                logger.info(f"Reached max frames ({args.max_frames}), stopping")
                break

    finally:
        cap.release()

    logger.info("-" * 50)
    logger.info(f"Saved {len(saved_frames)} annotated frames to {args.output_dir}/")
    logger.info(f"View with: open {args.output_dir}")


if __name__ == "__main__":
    main()
