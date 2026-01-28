# SAM3 Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace broken SAM2 tracking with SAM3's unified text-prompted video segmentation for stable player tracking.

**Architecture:** Single SAM3 VideoPredictor processes entire video with "basketball player" prompt, producing stable track IDs. Dedicated pipeline (not adapter) integrates with existing FrameDetections types and database storage.

**Tech Stack:** HuggingFace transformers (SAM3), PyTorch (MPS/CUDA), existing SmolVLM2 OCR

---

## Phase 1: SAM3 Core Integration

### Task 1.1: Add SAM3 Dependencies

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: Add transformers from git for MPS support**

Add to `backend/requirements.txt`:

```
# SAM3 via HuggingFace transformers (MPS compatible)
git+https://github.com/huggingface/transformers
```

**Step 2: Verify installation**

Run:
```bash
cd backend && pip install -r requirements.txt
```

Expected: Installation completes without errors

**Step 3: Test import**

Run:
```bash
cd backend && python -c "from transformers import Sam3Model; print('SAM3 import OK')"
```

Expected: "SAM3 import OK" (may need HuggingFace login first)

**Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "build: add SAM3 via HuggingFace transformers"
```

---

### Task 1.2: Add SAM3 Configuration

**Files:**
- Modify: `backend/app/config.py`

**Step 1: Write the test**

Create `backend/tests/test_sam3_config.py`:

```python
"""Tests for SAM3 configuration."""

import pytest
from pathlib import Path


class TestSAM3Config:
    """Tests for SAM3 configuration settings."""

    def test_sam3_settings_exist(self) -> None:
        """Test that SAM3 settings are defined in config."""
        from app.config import settings

        assert hasattr(settings, "sam3_prompt")
        assert hasattr(settings, "sam3_confidence_threshold")
        assert hasattr(settings, "sam3_use_half_precision")
        assert hasattr(settings, "sam3_temp_frames_dir")

    def test_sam3_default_values(self) -> None:
        """Test SAM3 default configuration values."""
        from app.config import settings

        assert settings.sam3_prompt == "basketball player"
        assert settings.sam3_confidence_threshold == 0.25
        assert settings.sam3_use_half_precision is True
        assert isinstance(settings.sam3_temp_frames_dir, Path)

    def test_tracking_backend_includes_sam3(self) -> None:
        """Test that sam3 is a valid tracking backend option."""
        from app.config import Settings

        # Check the type annotation includes sam3
        tracking_field = Settings.model_fields["tracking_backend"]
        assert "sam3" in str(tracking_field.annotation)
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_config.py -v
```

Expected: FAIL with "AttributeError: 'Settings' object has no attribute 'sam3_prompt'"

**Step 3: Add SAM3 settings to config.py**

Add after line 78 (after sam2_reidentification_enabled) in `backend/app/config.py`:

```python
    # SAM3 tracking settings
    sam3_prompt: str = "basketball player"
    sam3_confidence_threshold: float = 0.25
    sam3_use_half_precision: bool = True
    sam3_temp_frames_dir: Path = Path("/tmp/sam3_frames")
```

Also update `tracking_backend` type on line 28:

```python
    tracking_backend: Literal["bytetrack", "norfair", "sam2", "sam3"] = "norfair"
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_config.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/test_sam3_config.py
git commit -m "feat(config): add SAM3 tracking settings"
```

---

### Task 1.3: Create Frame Extractor for SAM3

SAM3 VideoPredictor requires frames as JPEGs on disk. Create utility to extract frames.

**Files:**
- Create: `backend/app/ml/sam3_frame_extractor.py`
- Create: `backend/tests/test_sam3_frame_extractor.py`

**Step 1: Write the failing test**

Create `backend/tests/test_sam3_frame_extractor.py`:

```python
"""Tests for SAM3 frame extractor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSAM3FrameExtractor:
    """Tests for SAM3 frame extraction utility."""

    def test_extract_frames_creates_directory(self, tmp_path: Path) -> None:
        """Test that extract_frames creates output directory."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"

        # Mock video capture to return empty
        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.read.return_value = (False, None)
            mock_cap.return_value.get.return_value = 30.0  # fps

            extractor.extract_frames(Path("fake_video.mp4"), output_dir)

        assert output_dir.exists()

    def test_extract_frames_respects_sample_interval(self, tmp_path: Path) -> None:
        """Test that extract_frames samples every Nth frame."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"

        # Create fake frames
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.return_value = 30.0

            # Return 10 frames then stop
            call_count = [0]
            def mock_read():
                call_count[0] += 1
                if call_count[0] <= 10:
                    return (True, fake_frame.copy())
                return (False, None)

            mock_cap.return_value.read.side_effect = mock_read

            with patch("app.ml.sam3_frame_extractor.cv2.imwrite") as mock_write:
                mock_write.return_value = True
                result = extractor.extract_frames(
                    Path("fake.mp4"), output_dir, sample_interval=3
                )

        # With 10 frames and interval=3, should get frames 0, 3, 6, 9 = 4 frames
        assert result.frame_count == 4
        assert result.sample_interval == 3

    def test_extract_frames_returns_metadata(self, tmp_path: Path) -> None:
        """Test that extract_frames returns correct metadata."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()
        output_dir = tmp_path / "frames"
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch("app.ml.sam3_frame_extractor.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = lambda prop: {
                5: 30.0,  # CAP_PROP_FPS
                3: 1280,  # CAP_PROP_FRAME_WIDTH
                4: 720,   # CAP_PROP_FRAME_HEIGHT
            }.get(prop, 0)

            call_count = [0]
            def mock_read():
                call_count[0] += 1
                if call_count[0] <= 3:
                    return (True, fake_frame.copy())
                return (False, None)

            mock_cap.return_value.read.side_effect = mock_read

            with patch("app.ml.sam3_frame_extractor.cv2.imwrite", return_value=True):
                result = extractor.extract_frames(Path("fake.mp4"), output_dir)

        assert result.fps == 30.0
        assert result.width == 1280
        assert result.height == 720
        assert result.output_dir == output_dir

    def test_context_manager_cleanup(self, tmp_path: Path) -> None:
        """Test that context manager cleans up temp directory."""
        from app.ml.sam3_frame_extractor import SAM3FrameExtractor

        extractor = SAM3FrameExtractor()

        with extractor.temp_frame_folder("test_video") as folder:
            temp_path = folder
            assert folder.exists()
            # Create a file to verify cleanup
            (folder / "test.jpg").touch()

        # After context, folder should be deleted
        assert not temp_path.exists()
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_frame_extractor.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.ml.sam3_frame_extractor'"

**Step 3: Implement SAM3FrameExtractor**

Create `backend/app/ml/sam3_frame_extractor.py`:

```python
"""Frame extraction utility for SAM3 VideoPredictor.

SAM3's VideoPredictor requires frames as JPEG files on disk.
This module handles extracting frames from video files with
configurable sampling and cleanup.
"""

import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractionResult:
    """Result of frame extraction operation."""

    output_dir: Path
    frame_count: int
    fps: float
    width: int
    height: int
    sample_interval: int
    frame_indices: list[int]  # Original frame numbers that were extracted


class SAM3FrameExtractor:
    """Extracts video frames to JPEG files for SAM3 processing."""

    def __init__(self, jpeg_quality: int = 95):
        """Initialize frame extractor.

        Args:
            jpeg_quality: JPEG compression quality (0-100).
        """
        self.jpeg_quality = jpeg_quality

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        sample_interval: int = 1,
    ) -> FrameExtractionResult:
        """Extract frames from video to JPEG files.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save JPEG frames.
            sample_interval: Extract every Nth frame (default: every frame).

        Returns:
            FrameExtractionResult with metadata about extracted frames.

        Raises:
            ValueError: If video cannot be opened.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_indices = []
            output_idx = 0
            input_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if input_idx % sample_interval == 0:
                    # SAM3 expects frames named as 6-digit numbers
                    frame_path = output_dir / f"{output_idx:06d}.jpg"
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    frame_indices.append(input_idx)
                    output_idx += 1

                input_idx += 1

            logger.info(
                f"Extracted {output_idx} frames from {video_path} "
                f"(sample_interval={sample_interval})"
            )

            return FrameExtractionResult(
                output_dir=output_dir,
                frame_count=output_idx,
                fps=fps,
                width=width,
                height=height,
                sample_interval=sample_interval,
                frame_indices=frame_indices,
            )

        finally:
            cap.release()

    @contextmanager
    def temp_frame_folder(
        self, video_id: str
    ) -> Generator[Path, None, None]:
        """Context manager for temporary frame folder with cleanup.

        Args:
            video_id: Unique identifier for the video (used in folder name).

        Yields:
            Path to temporary folder.
        """
        folder = settings.sam3_temp_frames_dir / video_id
        try:
            folder.mkdir(parents=True, exist_ok=True)
            yield folder
        finally:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
                logger.debug(f"Cleaned up temp frames: {folder}")
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_frame_extractor.py -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/sam3_frame_extractor.py backend/tests/test_sam3_frame_extractor.py
git commit -m "feat(ml): add SAM3 frame extractor utility"
```

---

### Task 1.4: Create SAM3 Tracker Core

**Files:**
- Create: `backend/app/ml/sam3_tracker.py`
- Create: `backend/tests/test_sam3_tracker.py`

**Step 1: Write the failing test**

Create `backend/tests/test_sam3_tracker.py`:

```python
"""Tests for SAM3 video tracker."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.ml.types import BoundingBox, Detection, FrameDetections


class TestSAM3TrackerConfig:
    """Tests for SAM3TrackerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from app.ml.sam3_tracker import SAM3TrackerConfig

        config = SAM3TrackerConfig()
        assert config.prompt == "basketball player"
        assert config.confidence_threshold == 0.25
        assert config.device == "auto"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        from app.ml.sam3_tracker import SAM3TrackerConfig

        config = SAM3TrackerConfig(
            prompt="player in white",
            confidence_threshold=0.5,
            device="cuda",
        )
        assert config.prompt == "player in white"
        assert config.confidence_threshold == 0.5
        assert config.device == "cuda"


class TestSAM3VideoTracker:
    """Tests for SAM3VideoTracker class."""

    def test_init_creates_tracker(self) -> None:
        """Test that initialization creates tracker instance."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig()
        tracker = SAM3VideoTracker(config)

        assert tracker._config == config
        assert tracker._predictor is None  # Lazy loaded

    def test_select_device_prefers_cuda(self) -> None:
        """Test device selection prefers CUDA when available."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "cuda"

    def test_select_device_falls_back_to_mps(self) -> None:
        """Test device selection falls back to MPS when CUDA unavailable."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.zeros.return_value = MagicMock()  # MPS test succeeds

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "mps"

    def test_select_device_falls_back_to_cpu(self) -> None:
        """Test device selection falls back to CPU as last resort."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        with patch("app.ml.sam3_tracker.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            config = SAM3TrackerConfig(device="auto")
            tracker = SAM3VideoTracker(config)
            device = tracker._select_device()

            assert device == "cpu"

    def test_convert_to_frame_detections(self) -> None:
        """Test conversion from SAM3 output to FrameDetections."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig()
        tracker = SAM3VideoTracker(config)

        # Mock SAM3 output format
        sam3_output = {
            "frame_index": 5,
            "object_ids": [1, 2, 3],
            "boxes": np.array([
                [10, 20, 100, 200],  # x1, y1, x2, y2
                [200, 100, 300, 250],
                [400, 150, 500, 350],
            ]),
            "scores": np.array([0.95, 0.88, 0.75]),
            "masks": [
                np.ones((200, 100), dtype=bool),
                np.ones((150, 100), dtype=bool),
                np.ones((200, 100), dtype=bool),
            ],
        }

        result = tracker._convert_to_frame_detections(
            sam3_output,
            frame_number=15,  # Original frame number (before sampling)
            frame_width=1920,
            frame_height=1080,
        )

        assert isinstance(result, FrameDetections)
        assert result.frame_number == 15
        assert len(result.detections) == 3
        assert result.detections[0].tracking_id == 1
        assert result.detections[1].tracking_id == 2
        assert result.detections[2].tracking_id == 3
        assert result.detections[0].confidence == 0.95

    def test_convert_filters_low_confidence(self) -> None:
        """Test that conversion filters out low confidence detections."""
        from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig

        config = SAM3TrackerConfig(confidence_threshold=0.5)
        tracker = SAM3VideoTracker(config)

        sam3_output = {
            "frame_index": 0,
            "object_ids": [1, 2],
            "boxes": np.array([[10, 20, 100, 200], [200, 100, 300, 250]]),
            "scores": np.array([0.8, 0.3]),  # Second one below threshold
            "masks": [np.ones((100, 100), dtype=bool)] * 2,
        }

        result = tracker._convert_to_frame_detections(
            sam3_output, frame_number=0, frame_width=1920, frame_height=1080
        )

        assert len(result.detections) == 1
        assert result.detections[0].tracking_id == 1
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_tracker.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'app.ml.sam3_tracker'"

**Step 3: Implement SAM3VideoTracker**

Create `backend/app/ml/sam3_tracker.py`:

```python
"""SAM3 video tracker using Meta's Segment Anything Model 3.

SAM3 provides unified detection, segmentation, and tracking with
text prompts. This module wraps SAM3's VideoPredictor for basketball
player tracking.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from app.config import settings

from .sam3_frame_extractor import SAM3FrameExtractor
from .types import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class SAM3TrackerConfig:
    """Configuration for SAM3 video tracker."""

    prompt: str = "basketball player"
    confidence_threshold: float = 0.25
    device: str = "auto"
    use_half_precision: bool = True


class SAM3VideoTracker:
    """SAM3-based tracker using text-prompted video segmentation.

    Uses SAM3's VideoPredictor to detect and track all instances of
    "basketball player" (or custom prompt) throughout a video with
    stable object IDs.

    Example:
        tracker = SAM3VideoTracker(SAM3TrackerConfig())
        for frame_detections in tracker.process_video(video_path):
            # Each detection has stable tracking_id
            print(frame_detections)
    """

    def __init__(self, config: SAM3TrackerConfig | None = None) -> None:
        """Initialize SAM3 tracker.

        Args:
            config: Tracker configuration. If None, uses defaults from settings.
        """
        self._config = config or SAM3TrackerConfig(
            prompt=settings.sam3_prompt,
            confidence_threshold=settings.sam3_confidence_threshold,
            use_half_precision=settings.sam3_use_half_precision,
        )

        # Lazy-loaded components
        self._predictor = None
        self._device = None
        self._frame_extractor = SAM3FrameExtractor()

        logger.info(
            f"SAM3VideoTracker initialized with prompt='{self._config.prompt}', "
            f"device={self._config.device}"
        )

    def _select_device(self) -> str:
        """Select best available device with fallback chain.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'.
        """
        import torch

        if self._config.device != "auto":
            return self._config.device

        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"

        if torch.backends.mps.is_available():
            # Test MPS actually works
            try:
                torch.zeros(1, device="mps")
                logger.info("Using MPS device")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS available but failed test: {e}")

        logger.info("Using CPU device")
        return "cpu"

    def _load_predictor(self) -> None:
        """Lazy load SAM3 VideoPredictor."""
        if self._predictor is not None:
            return

        try:
            from transformers import Sam3VideoPredictor

            self._device = self._select_device()

            logger.info(f"Loading SAM3 VideoPredictor on {self._device}...")

            self._predictor = Sam3VideoPredictor.from_pretrained(
                "facebook/sam3",
                device=self._device,
                torch_dtype="float16" if self._config.use_half_precision else "float32",
            )

            logger.info("SAM3 VideoPredictor loaded successfully")

        except ImportError as e:
            raise ImportError(
                "SAM3 not installed. Install with: "
                "pip install git+https://github.com/huggingface/transformers"
            ) from e

    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 3,
    ) -> Generator[FrameDetections, None, None]:
        """Process video and yield FrameDetections for each frame.

        Args:
            video_path: Path to input video file.
            sample_interval: Process every Nth frame.

        Yields:
            FrameDetections for each processed frame with stable tracking IDs.
        """
        self._load_predictor()

        video_id = video_path.stem

        with self._frame_extractor.temp_frame_folder(video_id) as frames_dir:
            # Extract frames to JPEG folder
            extraction = self._frame_extractor.extract_frames(
                video_path, frames_dir, sample_interval=sample_interval
            )

            logger.info(
                f"Processing {extraction.frame_count} frames with SAM3 "
                f"(prompt='{self._config.prompt}')"
            )

            # Start SAM3 session
            session_id = self._predictor.start_session(str(frames_dir))

            try:
                # Add text prompt at frame 0
                self._predictor.add_prompt(
                    session_id=session_id,
                    frame_index=0,
                    text=self._config.prompt,
                )

                # Propagate through video
                for output in self._predictor.propagate_in_video(
                    session_id=session_id,
                    direction="forward",
                ):
                    # Map SAM3 frame index back to original video frame number
                    sam3_frame_idx = output["frame_index"]
                    original_frame_number = extraction.frame_indices[sam3_frame_idx]

                    frame_detections = self._convert_to_frame_detections(
                        output,
                        frame_number=original_frame_number,
                        frame_width=extraction.width,
                        frame_height=extraction.height,
                    )

                    yield frame_detections

            finally:
                self._predictor.close_session(session_id)

    def _convert_to_frame_detections(
        self,
        sam3_output: dict,
        frame_number: int,
        frame_width: int,
        frame_height: int,
    ) -> FrameDetections:
        """Convert SAM3 output to FrameDetections type.

        Args:
            sam3_output: Output dict from SAM3 propagate_in_video.
            frame_number: Original video frame number.
            frame_width: Video frame width.
            frame_height: Video frame height.

        Returns:
            FrameDetections compatible with existing pipeline.
        """
        detections = []

        object_ids = sam3_output.get("object_ids", [])
        boxes = sam3_output.get("boxes", np.array([]))
        scores = sam3_output.get("scores", np.array([]))
        masks = sam3_output.get("masks", [])

        for i, obj_id in enumerate(object_ids):
            # Skip low confidence detections
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if confidence < self._config.confidence_threshold:
                continue

            # Convert box from x1y1x2y2 to BoundingBox
            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                bbox = BoundingBox.from_xyxy(float(x1), float(y1), float(x2), float(y2))
            else:
                continue

            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=0,  # person
                class_name="person",
                tracking_id=int(obj_id),
            )
            detections.append(detection)

        return FrameDetections(
            frame_number=frame_number,
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_tracker.py -v
```

Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add backend/app/ml/sam3_tracker.py backend/tests/test_sam3_tracker.py
git commit -m "feat(ml): add SAM3 video tracker core"
```

---

### Task 1.5: Integration Test with Real Video

**Files:**
- Create: `backend/scripts/test_sam3_tracking.py`

**Step 1: Create integration test script**

Create `backend/scripts/test_sam3_tracking.py`:

```python
#!/usr/bin/env python
"""Integration test for SAM3 video tracking.

Usage:
    python -m scripts.test_sam3_tracking <video_path>

This script tests SAM3 tracking on a real video and reports:
- Number of unique track IDs (should be ~10 for basketball, not 50+)
- Track stability (same player should keep same ID)
- Processing time
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
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
    track_id_per_frame = {}

    for frame_detections in tracker.process_video(
        args.video_path, sample_interval=args.sample_interval
    ):
        frame_count += 1
        frame_track_ids = [d.tracking_id for d in frame_detections.detections]
        all_track_ids.extend(frame_track_ids)
        track_id_per_frame[frame_detections.frame_number] = frame_track_ids

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
    logger.info(f"Processing time: {elapsed:.1f}s ({elapsed/frame_count:.2f}s/frame)")
    logger.info(f"Unique track IDs: {len(unique_ids)}")
    logger.info(f"Total detections: {len(all_track_ids)}")

    logger.info("\nTrack ID frequency (top 15):")
    for track_id, count in id_counts.most_common(15):
        logger.info(f"  Track {track_id}: {count} detections")

    # Success criteria
    if len(unique_ids) <= 15:
        logger.info("\n✅ PASS: Reasonable number of track IDs")
    else:
        logger.warning(f"\n⚠️  WARN: {len(unique_ids)} track IDs (expected ~10)")


if __name__ == "__main__":
    main()
```

**Step 2: Make script executable and test**

Run:
```bash
chmod +x backend/scripts/test_sam3_tracking.py
```

**Step 3: Test with a real video (manual)**

Run when you have a test video:
```bash
cd backend && python -m scripts.test_sam3_tracking /path/to/basketball/video.mp4
```

Expected: Output showing track IDs and processing metrics

**Step 4: Commit**

```bash
git add backend/scripts/test_sam3_tracking.py
git commit -m "feat(scripts): add SAM3 tracking integration test"
```

---

## Phase 2: Pipeline Integration

### Task 2.1: Create SAM3 Detection Pipeline

**Files:**
- Create: `backend/app/services/sam3_detection_pipeline.py`
- Create: `backend/tests/test_sam3_detection_pipeline.py`

**Step 1: Write the failing test**

Create `backend/tests/test_sam3_detection_pipeline.py`:

```python
"""Tests for SAM3 detection pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ml.types import BoundingBox, Detection, FrameDetections


class TestSAM3DetectionPipeline:
    """Tests for SAM3DetectionPipeline class."""

    def test_init_creates_pipeline(self) -> None:
        """Test that initialization creates pipeline instance."""
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        pipeline = SAM3DetectionPipeline()
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_process_video_yields_detections(self) -> None:
        """Test that process_video yields FrameDetections."""
        from app.services.sam3_detection_pipeline import SAM3DetectionPipeline

        pipeline = SAM3DetectionPipeline()

        # Mock the tracker
        mock_detections = [
            FrameDetections(
                frame_number=0,
                detections=[
                    Detection(
                        bbox=BoundingBox(10, 20, 100, 200),
                        confidence=0.9,
                        class_id=0,
                        class_name="person",
                        tracking_id=1,
                    )
                ],
                frame_width=1920,
                frame_height=1080,
            )
        ]

        with patch.object(
            pipeline, "_tracker"
        ) as mock_tracker:
            mock_tracker.process_video.return_value = iter(mock_detections)

            results = []
            async for frame_det in pipeline.process_video(Path("fake.mp4")):
                results.append(frame_det)

            assert len(results) == 1
            assert results[0].frame_number == 0
            assert results[0].detections[0].tracking_id == 1
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_detection_pipeline.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement SAM3DetectionPipeline**

Create `backend/app/services/sam3_detection_pipeline.py`:

```python
"""SAM3-based detection pipeline for video processing.

This pipeline uses SAM3's unified detection and tracking to process
basketball videos with stable player tracking.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator, Callable

from app.config import settings
from app.ml.sam3_tracker import SAM3TrackerConfig, SAM3VideoTracker
from app.ml.types import FrameDetections

logger = logging.getLogger(__name__)


class SAM3DetectionPipeline:
    """Pipeline for processing videos with SAM3 tracking.

    Replaces the traditional detect -> track pipeline with SAM3's
    unified text-prompted video segmentation.
    """

    def __init__(
        self,
        prompt: str | None = None,
        confidence_threshold: float | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize SAM3 detection pipeline.

        Args:
            prompt: Text prompt for detection (default: from settings).
            confidence_threshold: Min confidence (default: from settings).
            on_progress: Optional callback(current_frame, total_frames).
        """
        config = SAM3TrackerConfig(
            prompt=prompt or settings.sam3_prompt,
            confidence_threshold=confidence_threshold
            or settings.sam3_confidence_threshold,
        )
        self._tracker = SAM3VideoTracker(config)
        self._on_progress = on_progress

    async def process_video(
        self,
        video_path: Path,
        sample_interval: int | None = None,
    ) -> AsyncGenerator[FrameDetections, None]:
        """Process video and yield FrameDetections.

        Args:
            video_path: Path to video file.
            sample_interval: Process every Nth frame (default: from settings).

        Yields:
            FrameDetections for each processed frame.
        """
        interval = sample_interval or settings.batch_sample_interval

        logger.info(f"Starting SAM3 pipeline for {video_path}")

        frame_count = 0
        for frame_detections in self._tracker.process_video(
            video_path, sample_interval=interval
        ):
            frame_count += 1

            if self._on_progress:
                self._on_progress(frame_count, -1)  # Total unknown

            yield frame_detections

        logger.info(f"SAM3 pipeline complete: {frame_count} frames processed")

    async def process_video_to_db(
        self,
        video_id: int,
        video_path: Path,
        db_session,
        sample_interval: int | None = None,
    ) -> int:
        """Process video and store detections in database.

        Args:
            video_id: Database video ID.
            video_path: Path to video file.
            db_session: Database session for storage.
            sample_interval: Process every Nth frame.

        Returns:
            Number of frames processed.
        """
        from app.models import PlayerDetection

        frame_count = 0

        async for frame_detections in self.process_video(
            video_path, sample_interval=sample_interval
        ):
            # Store each detection in database
            for det in frame_detections.detections:
                detection = PlayerDetection(
                    video_id=video_id,
                    frame_number=frame_detections.frame_number,
                    tracking_id=det.tracking_id,
                    bbox_x=det.bbox.x,
                    bbox_y=det.bbox.y,
                    bbox_width=det.bbox.width,
                    bbox_height=det.bbox.height,
                    confidence=det.confidence,
                )
                db_session.add(detection)

            frame_count += 1

            # Commit every 100 frames for checkpointing
            if frame_count % 100 == 0:
                await db_session.commit()
                logger.debug(f"Checkpointed at frame {frame_count}")

        await db_session.commit()
        return frame_count
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_detection_pipeline.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add backend/app/services/sam3_detection_pipeline.py backend/tests/test_sam3_detection_pipeline.py
git commit -m "feat(services): add SAM3 detection pipeline"
```

---

### Task 2.2: Wire Up API Endpoint

**Files:**
- Modify: `backend/app/api/detection.py`

**Step 1: Add SAM3 backend option to detection API**

This task involves modifying existing code. Read the file first, then add SAM3 support.

Run to see current structure:
```bash
cd backend && head -100 app/api/detection.py
```

Add conditional logic to use SAM3 pipeline when `tracking_backend == "sam3"`.

**Step 2: Test via API**

```bash
# Start server
cd backend && uvicorn app.main:app --reload

# Trigger detection (in another terminal)
curl -X POST http://localhost:8000/api/detection/videos/1/detect
```

**Step 3: Commit**

```bash
git add backend/app/api/detection.py
git commit -m "feat(api): add SAM3 backend support to detection endpoint"
```

---

## Phase 3: Jersey OCR Integration

### Task 3.1: Add Mask-Based Jersey Crop Extraction

**Files:**
- Create: `backend/app/ml/sam3_jersey_extractor.py`
- Create: `backend/tests/test_sam3_jersey_extractor.py`

**Step 1: Write the failing test**

Create `backend/tests/test_sam3_jersey_extractor.py`:

```python
"""Tests for SAM3 mask-based jersey extraction."""

import numpy as np
import pytest

from app.ml.types import BoundingBox


class TestSAM3JerseyExtractor:
    """Tests for mask-based jersey crop extraction."""

    def test_extract_jersey_crop_uses_mask(self) -> None:
        """Test that jersey extraction uses mask to remove background."""
        from app.ml.sam3_jersey_extractor import extract_jersey_crop

        # Create test frame (100x100 RGB)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Create mask (only center is player)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        bbox = BoundingBox(x=10, y=10, width=80, height=80)

        crop = extract_jersey_crop(frame, bbox, mask=mask)

        # Crop should be upper 40% of bbox height
        expected_height = int(80 * 0.4)  # 32 pixels
        assert crop.shape[0] == expected_height

        # Background pixels (outside mask) should be zeroed
        # Check corners which are outside mask
        assert crop[0, 0, 0] == 0  # Top-left corner

    def test_extract_jersey_crop_without_mask(self) -> None:
        """Test jersey extraction without mask (fallback)."""
        from app.ml.sam3_jersey_extractor import extract_jersey_crop

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=80, height=80)

        crop = extract_jersey_crop(frame, bbox, mask=None)

        # Should still return upper 40%
        expected_height = int(80 * 0.4)
        assert crop.shape[0] == expected_height
        # Without mask, pixels should be unchanged
        assert crop[0, 0, 0] == 128
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_jersey_extractor.py -v
```

**Step 3: Implement extract_jersey_crop**

Create `backend/app/ml/sam3_jersey_extractor.py`:

```python
"""Jersey crop extraction using SAM3 masks.

SAM3 provides precise segmentation masks that can be used to
extract jersey regions while removing background pixels.
"""

import numpy as np

from .types import BoundingBox


def extract_jersey_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    mask: np.ndarray | None = None,
    jersey_height_ratio: float = 0.4,
) -> np.ndarray:
    """Extract jersey region from frame using optional mask.

    Args:
        frame: BGR frame image.
        bbox: Player bounding box.
        mask: Optional segmentation mask (same size as frame).
        jersey_height_ratio: Fraction of bbox height for jersey (default: 40%).

    Returns:
        Cropped jersey region with background zeroed if mask provided.
    """
    x1, y1, x2, y2 = bbox.to_xyxy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Jersey is upper portion of player
    jersey_y2 = y1 + int((y2 - y1) * jersey_height_ratio)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    jersey_y2 = min(frame.shape[0], jersey_y2)

    # Extract crop
    crop = frame[y1:jersey_y2, x1:x2].copy()

    # Apply mask if provided
    if mask is not None:
        crop_mask = mask[y1:jersey_y2, x1:x2]
        # Zero out background pixels
        crop[~crop_mask] = 0

    return crop
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_sam3_jersey_extractor.py -v
```

**Step 5: Commit**

```bash
git add backend/app/ml/sam3_jersey_extractor.py backend/tests/test_sam3_jersey_extractor.py
git commit -m "feat(ml): add mask-based jersey crop extraction for SAM3"
```

---

## Phase 4: Polish & Cleanup

### Task 4.1: Add Progress Logging

Modify `sam3_tracker.py` to emit progress logs during processing.

### Task 4.2: Test on Real Video

Run full integration test with actual basketball video.

### Task 4.3: Update Documentation

Update `docs/architecture.md` to reflect SAM3 pipeline option.

### Task 4.4: Clean Up Old Code (Optional)

Consider deprecating or removing `sam2_tracker.py` once SAM3 is validated.

---

## Success Criteria Checklist

- [ ] SAM3 installs successfully via HuggingFace transformers
- [ ] Device selection works (CUDA > MPS > CPU)
- [ ] Frame extraction creates proper JPEG folder
- [ ] SAM3 VideoPredictor processes video with text prompt
- [ ] Track IDs remain stable (same player = same ID)
- [ ] ~10 track IDs per video (not 50+)
- [ ] Jersey OCR can use SAM3 masks for better crops
- [ ] Processing time < 5 min for 10 min video

---

## Quick Reference

**Run all SAM3 tests:**
```bash
cd backend && python -m pytest tests/test_sam3*.py -v
```

**Test tracking on video:**
```bash
cd backend && python -m scripts.test_sam3_tracking /path/to/video.mp4
```

**Check device selection:**
```bash
cd backend && python -c "from app.ml.sam3_tracker import SAM3VideoTracker, SAM3TrackerConfig; t = SAM3VideoTracker(SAM3TrackerConfig()); print(t._select_device())"
```
