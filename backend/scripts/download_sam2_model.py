#!/usr/bin/env python3
"""Download SAM2 model checkpoints.

Usage:
    python -m scripts.download_sam2_model [model_name]

Models available:
    - sam2_hiera_tiny (default, ~150MB)
    - sam2_hiera_small (~180MB)
    - sam2_hiera_base_plus (~320MB)
    - sam2_hiera_large (~900MB)
"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

# SAM2 model URLs from Meta's official repository
SAM2_MODELS = {
    "sam2_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "size_mb": 150,
    },
    "sam2_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "size_mb": 180,
    },
    "sam2_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "size_mb": 320,
    },
    "sam2_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "size_mb": 900,
    },
}


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\rDownloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def download_model(model_name: str, models_dir: Path) -> Path:
    """Download SAM2 model checkpoint.

    Args:
        model_name: Name of the model to download.
        models_dir: Directory to save the model.

    Returns:
        Path to the downloaded model file.
    """
    if model_name not in SAM2_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(SAM2_MODELS.keys())}")
        sys.exit(1)

    model_info = SAM2_MODELS[model_name]
    url = model_info["url"]
    filename = f"{model_name}.pt"
    filepath = models_dir / filename

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if filepath.exists():
        print(f"Model already exists: {filepath}")
        return filepath

    print(f"Downloading {model_name} (~{model_info['size_mb']}MB)...")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")

    try:
        urlretrieve(url, filepath, reporthook=download_progress)
        print()  # New line after progress
        print(f"Downloaded successfully: {filepath}")
        return filepath
    except Exception as e:
        print(f"\nError downloading model: {e}")
        if filepath.exists():
            filepath.unlink()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download SAM2 model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  sam2_hiera_tiny       Smallest model (~150MB, fastest)
  sam2_hiera_small      Small model (~180MB)
  sam2_hiera_base_plus  Base model (~320MB)
  sam2_hiera_large      Large model (~900MB, best quality)
        """,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="sam2_hiera_tiny",
        choices=list(SAM2_MODELS.keys()),
        help="Model to download (default: sam2_hiera_tiny)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory to save models (default: ./models)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )

    args = parser.parse_args()

    if args.all:
        for model_name in SAM2_MODELS:
            download_model(model_name, args.models_dir)
    else:
        download_model(args.model, args.models_dir)


if __name__ == "__main__":
    main()
