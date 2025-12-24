#!/usr/bin/env python3
"""Benchmark ML inference performance across different devices.

Compares YOLO detection speed on CPU, MPS (Apple Silicon), and CUDA (NVIDIA).
Helps determine optimal batch sizes and expected throughput for each device.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.yolo_detector import YOLODetector


def generate_test_frames(num_frames: int = 100, width: int = 1920, height: int = 1080) -> list[np.ndarray]:
    """Generate synthetic test frames for benchmarking.

    Args:
        num_frames: Number of frames to generate
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        List of random BGR frames
    """
    print(f"Generating {num_frames} test frames ({width}x{height})...")
    frames = []
    for _ in range(num_frames):
        # Generate random noise frame (BGR format)
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def benchmark_device(
    device: str,
    batch_sizes: list[int],
    frames: list[np.ndarray],
    confidence_threshold: float = 0.5,
) -> dict:
    """Benchmark YOLO detection on a specific device.

    Args:
        device: Device to test ('cpu', 'mps', 'cuda')
        batch_sizes: List of batch sizes to test
        frames: Test frames to process
        confidence_threshold: Detection confidence threshold

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking device: {device.upper()}")
    print(f"{'='*60}")

    results = {
        "device": device,
        "batch_results": [],
        "total_frames": len(frames),
    }

    try:
        # Initialize detector
        detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # Warm up model (first run is slower due to model loading)
        print("Warming up model...")
        _ = detector.detect_batch(frames[:8], 0)

        # Test each batch size
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}...")

            num_batches = len(frames) // batch_size
            frames_to_process = num_batches * batch_size  # Only complete batches

            start_time = time.time()

            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch = frames[batch_start:batch_end]

                _ = detector.detect_batch(batch, batch_start)

            elapsed_time = time.time() - start_time
            fps = frames_to_process / elapsed_time

            batch_result = {
                "batch_size": batch_size,
                "frames_processed": frames_to_process,
                "elapsed_time": elapsed_time,
                "fps": fps,
                "avg_batch_time": elapsed_time / num_batches,
            }
            results["batch_results"].append(batch_result)

            print(f"  Processed {frames_to_process} frames in {elapsed_time:.2f}s")
            print(f"  Throughput: {fps:.1f} FPS")
            print(f"  Avg batch time: {batch_result['avg_batch_time']:.3f}s")

    except Exception as e:
        print(f"  ERROR: {e}")
        results["error"] = str(e)

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print comparison summary across all devices.

    Args:
        all_results: List of results from each device
    """
    print(f"\n\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")

    # Print table header
    print(f"{'Device':<10} {'Batch Size':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 42)

    # Collect all results by batch size
    baseline_fps = {}  # CPU results as baseline

    for result in all_results:
        if "error" in result:
            print(f"{result['device']:<10} ERROR: {result['error']}")
            continue

        for batch_result in result["batch_results"]:
            batch_size = batch_result["batch_size"]
            fps = batch_result["fps"]

            # Store CPU as baseline
            if result["device"] == "cpu":
                baseline_fps[batch_size] = fps
                speedup = "1.0x"
            else:
                # Calculate speedup relative to CPU
                cpu_baseline = baseline_fps.get(batch_size, fps)
                speedup_val = fps / cpu_baseline if cpu_baseline > 0 else 0
                speedup = f"{speedup_val:.1f}x"

            print(f"{result['device']:<10} {batch_size:<12} {fps:<10.1f} {speedup:<10}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find optimal batch sizes for each device
    for result in all_results:
        if "error" in result:
            continue

        best_batch = max(result["batch_results"], key=lambda x: x["fps"])
        print(f"\n{result['device'].upper()}:")
        print(f"  Optimal batch size: {best_batch['batch_size']}")
        print(f"  Max throughput: {best_batch['fps']:.1f} FPS")
        print(f"  Avg batch time: {best_batch['avg_batch_time']:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML inference across devices")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "mps"],
        help="Devices to test (cpu, mps, cuda)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of test frames to generate",
    )
    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="Frame resolution (WIDTHxHEIGHT)",
    )

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Generate test frames
    frames = generate_test_frames(args.num_frames, width, height)

    # Benchmark each device
    all_results = []
    for device in args.devices:
        result = benchmark_device(device, args.batch_sizes, frames)
        all_results.append(result)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
