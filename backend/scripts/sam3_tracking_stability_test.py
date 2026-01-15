#!/usr/bin/env python3
"""Test SAM3 tracking stability with tight frame intervals.

This script tests tracking ID consistency over consecutive frames
to measure how often track IDs switch between players.

Usage:
    uv run python -m scripts.sam3_tracking_stability_test <video_path>
"""

import argparse
import logging
import sys
from collections import defaultdict
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


def iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def analyze_track_stability(frames_data: list) -> dict:
    """Analyze tracking stability across frames.

    Returns metrics about:
    - Total unique IDs
    - Average detections per frame
    - ID switches (when same spatial location gets different ID)
    """
    if len(frames_data) < 2:
        return {"error": "Need at least 2 frames"}

    all_ids = set()
    id_switches = 0
    detections_per_frame = []

    for i, frame in enumerate(frames_data):
        detections_per_frame.append(len(frame["detections"]))
        for det in frame["detections"]:
            all_ids.add(det.tracking_id)

        if i == 0:
            continue

        # Check for ID switches: find boxes that overlap significantly
        # between consecutive frames but have different IDs
        prev_frame = frames_data[i - 1]
        curr_frame = frame

        for prev_det in prev_frame["detections"]:
            best_iou = 0
            best_match = None
            for curr_det in curr_frame["detections"]:
                overlap = iou(prev_det.bbox, curr_det.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_match = curr_det

            # If high overlap but different ID, it's a switch
            if best_iou > 0.3 and best_match and best_match.tracking_id != prev_det.tracking_id:
                id_switches += 1
                logger.debug(
                    f"ID switch at frame {frame['frame_number']}: "
                    f"{prev_det.tracking_id} -> {best_match.tracking_id} (IoU={best_iou:.2f})"
                )

    return {
        "total_frames": len(frames_data),
        "unique_ids": len(all_ids),
        "id_switches": id_switches,
        "avg_detections": np.mean(detections_per_frame),
        "all_ids": sorted(all_ids),
    }


def draw_detections(frame: np.ndarray, detections: list, frame_number: int,
                    prompt: str, threshold: float, metrics: dict = None) -> np.ndarray:
    """Draw bounding boxes and tracking IDs on frame."""
    annotated = frame.copy()

    for det in detections:
        color = get_color(det.tracking_id)

        x1, y1 = int(det.bbox.x), int(det.bbox.y)
        x2, y2 = int(det.bbox.x + det.bbox.width), int(det.bbox.y + det.bbox.height)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw tracking ID label
        label = f"ID:{det.tracking_id} ({det.confidence:.2f})"
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

    if metrics:
        info3 = f"Unique IDs: {metrics.get('unique_ids', '?')} | Switches: {metrics.get('id_switches', '?')}"
        cv2.putText(annotated, info3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return annotated


def run_tracking_test(video_path: Path, prompt: str, threshold: float,
                      output_dir: Path, sample_interval: int, max_frames: int,
                      start_frame: int = 0):
    """Run SAM3 tracking and measure stability."""
    from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

    config_name = f"{prompt.replace(' ', '_')}_t{threshold}_s{sample_interval}"
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: prompt='{prompt}', threshold={threshold}")
    logger.info(f"Sample interval: {sample_interval}, max frames: {max_frames}")
    logger.info(f"Start frame: {start_frame}")
    logger.info(f"Output: {config_dir}")
    logger.info(f"{'='*60}")

    config = SAM3TrackerConfig(prompt=prompt, confidence_threshold=threshold)
    tracker = SAM3VideoTracker(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video")
        return None

    frames_data = []
    frame_count = 0

    try:
        for frame_detections in tracker.process_video(
            video_path,
            sample_interval=sample_interval,
            max_frames=max_frames,
            start_frame=start_frame,
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_detections.frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            frames_data.append({
                "frame_number": frame_detections.frame_number,
                "detections": frame_detections.detections,
            })

            frame_count += 1
            logger.info(f"  Frame {frame_detections.frame_number}: {len(frame_detections.detections)} detections")

    finally:
        cap.release()

    # Analyze tracking stability
    metrics = analyze_track_stability(frames_data)

    logger.info(f"\n--- Tracking Stability Analysis ---")
    logger.info(f"Total frames: {metrics.get('total_frames', 0)}")
    logger.info(f"Unique track IDs: {metrics.get('unique_ids', 0)}")
    logger.info(f"ID switches detected: {metrics.get('id_switches', 0)}")
    logger.info(f"Avg detections/frame: {metrics.get('avg_detections', 0):.1f}")
    logger.info(f"Track IDs: {metrics.get('all_ids', [])}")

    # Save annotated frames
    cap = cv2.VideoCapture(str(video_path))
    for frame_info in frames_data:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info["frame_number"])
        ret, frame = cap.read()
        if not ret:
            continue

        annotated = draw_detections(
            frame, frame_info["detections"],
            frame_info["frame_number"], prompt, threshold, metrics
        )
        output_path = config_dir / f"frame_{frame_info['frame_number']:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)
    cap.release()

    logger.info(f"\nSaved {len(frames_data)} annotated frames to {config_dir}")

    return {
        "prompt": prompt,
        "threshold": threshold,
        "sample_interval": sample_interval,
        **metrics,
        "output_dir": str(config_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 tracking stability")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument("--output-dir", type=Path, default=Path("sam3_stability"),
                       help="Output directory")
    parser.add_argument("--sample-interval", type=int, default=3,
                       help="Sample every Nth frame (default: 3 for tight tracking)")
    parser.add_argument("--max-frames", type=int, default=30,
                       help="Max frames to process")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Start from this frame number")
    parser.add_argument("--prompt", type=str, default="basketball player",
                       help="Text prompt for detection")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Confidence threshold")
    args = parser.parse_args()

    if not args.video_path.exists():
        logger.error(f"Video not found: {args.video_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_tracking_test(
        args.video_path,
        args.prompt,
        args.threshold,
        args.output_dir,
        args.sample_interval,
        args.max_frames,
        args.start_frame,
    )

    if result:
        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Prompt: {result['prompt']}")
        logger.info(f"Threshold: {result['threshold']}")
        logger.info(f"Sample interval: {result['sample_interval']}")
        logger.info(f"Unique IDs: {result['unique_ids']}")
        logger.info(f"ID switches: {result['id_switches']}")

        # Quality assessment
        if result['unique_ids'] <= 12 and result['id_switches'] <= 2:
            logger.info("QUALITY: GOOD - Stable tracking")
        elif result['id_switches'] <= 5:
            logger.info("QUALITY: MODERATE - Some tracking issues")
        else:
            logger.info("QUALITY: POOR - Unstable tracking, IDs switching frequently")


if __name__ == "__main__":
    main()
