#!/usr/bin/env python3
"""Test script to verify GPU usage during YOLO inference."""

import time
import numpy as np
import torch
from ultralytics import YOLO

print("="*70)
print("GPU Usage Test for YOLO Inference")
print("="*70)

# Check device availability
print(f"\nPyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Load model
print("\n--- Loading YOLO model ---")
model = YOLO('yolov8n.pt')
print(f"Initial model device: {model.device}")

# Move model to MPS
print("\n--- Moving model to MPS ---")
model.to('mps')
print(f"Model device after .to('mps'): {model.device}")

# Check model parameters
if hasattr(model, 'model') and model.model is not None:
    first_param = next(model.model.parameters())
    print(f"First parameter device: {first_param.device}")
    print(f"First parameter dtype: {first_param.dtype}")

# Create dummy frames (simulate batch processing)
print("\n--- Creating test frames ---")
batch_size = 16
frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
print(f"Created {len(frames)} dummy frames")

# Test 1: Inference with device parameter (what our code does)
print("\n--- Test 1: Inference with device='mps' parameter ---")
start = time.time()
results = model.predict(source=frames, device='mps', conf=0.5, verbose=False)
elapsed = time.time() - start
fps = len(frames) / elapsed
print(f"Processed {len(frames)} frames in {elapsed:.3f}s ({fps:.1f} fps)")

# Test 2: Check if MPS is actually being used by checking memory
print("\n--- Test 2: Check MPS memory usage ---")
if hasattr(torch.mps, 'current_allocated_memory'):
    try:
        allocated = torch.mps.current_allocated_memory() / (1024 ** 2)  # Convert to MB
        print(f"MPS memory allocated: {allocated:.2f} MB")
        if allocated > 0:
            print("✅ MPS is being used (memory allocated)")
        else:
            print("❌ No MPS memory allocated - likely using CPU!")
    except:
        print("Could not check MPS memory")
else:
    print("MPS memory tracking not available in this PyTorch version")

# Test 3: CPU baseline for comparison
print("\n--- Test 3: CPU baseline ---")
model.to('cpu')
start = time.time()
results_cpu = model.predict(source=frames, device='cpu', conf=0.5, verbose=False)
elapsed_cpu = time.time() - start
fps_cpu = len(frames) / elapsed_cpu
print(f"CPU: Processed {len(frames)} frames in {elapsed_cpu:.3f}s ({fps_cpu:.1f} fps)")

# Compare
print("\n--- Performance Comparison ---")
speedup = elapsed_cpu / elapsed
print(f"MPS: {fps:.1f} fps")
print(f"CPU: {fps_cpu:.1f} fps")
print(f"Speedup: {speedup:.2f}x")

if speedup < 1.5:
    print("\n⚠️  WARNING: MPS is not providing expected speedup!")
    print("Expected 2-3x speedup, but got {:.2f}x".format(speedup))
    print("This suggests MPS may not actually be used for inference.")
else:
    print("\n✅ MPS appears to be working correctly")

print("\n" + "="*70)
