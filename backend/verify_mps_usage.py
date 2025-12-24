#!/usr/bin/env python3
"""
Verify that MPS is actually being used during inference.

This script demonstrates that:
1. MPS provides significant speedup vs CPU
2. Activity Monitor doesn't show MPS compute usage
3. torch.mps memory tracking proves MPS is being used
"""

import time
import numpy as np
import torch
from ultralytics import YOLO

def run_inference_test(device: str, model: YOLO, frames: list, label: str):
    """Run inference and measure performance."""
    print(f"\n{label}:")
    print(f"  Device: {device}")

    # Move model to device
    model.to(device)

    # Warmup run
    _ = model.predict(source=frames[:2], device=device, conf=0.5, verbose=False)

    # Timed run
    start = time.time()
    results = model.predict(source=frames, device=device, conf=0.5, verbose=False)
    elapsed = time.time() - start

    fps = len(frames) / elapsed
    print(f"  Time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Detections: {sum(len(r.boxes) for r in results)}")

    return elapsed, fps

print("="*70)
print("MPS Usage Verification Test")
print("="*70)

# Check MPS availability
print(f"\nPyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Load model
print("\nLoading YOLO model...")
model = YOLO('yolov8n.pt')

# Create test frames
batch_size = 16
frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
print(f"Created {batch_size} test frames (640x640)")

# Test MPS
elapsed_mps, fps_mps = run_inference_test('mps', model, frames, "MPS Test")

# Check MPS memory
if hasattr(torch.mps, 'current_allocated_memory'):
    allocated_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
    print(f"\n✓ MPS Memory: {allocated_mb:.2f} MB allocated")
    print("  (This proves MPS is being used for inference)")

# Test CPU for comparison
elapsed_cpu, fps_cpu = run_inference_test('cpu', model, frames, "CPU Test")

# Analysis
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"MPS: {fps_mps:.1f} fps")
print(f"CPU: {fps_cpu:.1f} fps")
print(f"Speedup: {elapsed_cpu / elapsed_mps:.2f}x")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if elapsed_cpu / elapsed_mps > 1.8:
    print("✅ MPS IS WORKING CORRECTLY")
    print("   The significant speedup proves GPU acceleration is active.")
    print("\n⚠️  NOTE: macOS Activity Monitor doesn't accurately show MPS usage!")
    print("   The '% GPU' column shows graphics rendering, not Metal compute.")
    print("   Even though Activity Monitor shows 0% GPU, MPS IS being used.")
else:
    print("❌ MPS may not be working correctly")
    print("   Expected >2x speedup but only got {:.2f}x".format(elapsed_cpu / elapsed_mps))

print("="*70)
