#!/usr/bin/env python3
"""Check which device will be used for ML inference."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.detection_pipeline import DetectionPipeline

print("="*60)
print("Device Detection Check")
print("="*60)

# Check ML device setting
print(f"\nML_DEVICE setting: {settings.ml_device}")

# Resolve device
resolved_device = DetectionPipeline._resolve_device(settings.ml_device)
print(f"Resolved device: {resolved_device}")

# Check batch size
batch_size = DetectionPipeline._get_optimal_batch_size(resolved_device)
print(f"Optimal batch size: {batch_size}")

# Check PyTorch availability
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"MPS built: {torch.backends.mps.is_built()}")
    else:
        print("MPS backend not available (PyTorch < 1.12)")
except ImportError:
    print("\nPyTorch not installed!")

print("\n" + "="*60)
