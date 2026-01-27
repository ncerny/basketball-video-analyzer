"""SAM3 video tracker using Meta's Segment Anything Model 3.

SAM3 provides unified detection, segmentation, and tracking with
text prompts. This module wraps SAM3's VideoModel for basketball
player tracking.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch

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
    # Note: bfloat16 is NOT supported on MPS (Apple Silicon). When device is MPS,
    # the tracker will automatically use float32 regardless of this setting.
    use_half_precision: bool = True

    # Memory management strategy:
    #
    # Options:
    #   - 0 (default): Keep all frames in memory. Provides stable tracking IDs
    #     but memory grows with video length. Recommended for most use cases.
    #
    #   - >0: Rolling window. Prunes frame outputs older than this many frames
    #     to bound memory usage. Pruning only starts after processing this many
    #     frames. The impact on tracking ID stability after pruning starts is
    #     not fully characterized - use with caution for long videos.
    #
    # For long videos where OOM is a concern, consider:
    #   1. Using sample_interval > 1
    #   2. Using a rolling window (e.g., 200-500 frames)
    #   3. Running on a system with more memory
    memory_window_size: int = 0

    # torch.compile() optimization - can provide 10-30% speedup on modern GPUs
    # Disabled by default as it adds startup latency for compilation
    use_torch_compile: bool = False


class SAM3VideoTracker:
    """SAM3-based tracker using text-prompted video segmentation.

    Uses SAM3's VideoModel to detect and track all instances of
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
            memory_window_size=settings.sam3_memory_window_size,
            use_torch_compile=settings.sam3_use_torch_compile,
        )

        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._frame_extractor = SAM3FrameExtractor()

        logger.info(
            f"SAM3VideoTracker initialized with prompt='{self._config.prompt}', "
            f"device={self._config.device}"
        )

    def _find_model_path(self) -> str:
        """Find SAM3 model path, checking local paths first.

        Returns:
            Path to model (local path or HuggingFace model ID).
        """
        # Check paths in order of preference
        local_paths = [
            os.environ.get("CLOUD_MODEL_PATH", ""),  # Docker container
            "/models/sam3",  # Docker default
            str(Path.home() / ".cache/huggingface/hub/models--facebook--sam3/snapshots"),
        ]

        for path in local_paths:
            if path and Path(path).exists():
                # For HF cache, find the actual snapshot
                if "snapshots" in path:
                    try:
                        # Sort by modification time (newest first) for deterministic selection
                        snapshots = sorted(
                            Path(path).iterdir(),
                            key=lambda p: p.stat().st_mtime_ns,
                            reverse=True,
                        )
                        if snapshots:
                            return str(snapshots[0])
                    except (PermissionError, OSError) as e:
                        logger.debug(
                            f"Failed to traverse HF cache directory {path}: {e}. "
                            "Trying next path..."
                        )
                        continue
                return path

        # Fall back to HuggingFace download
        return "facebook/sam3"

    def _select_device(self) -> str:
        """Select best available device with fallback chain.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'.

        Raises:
            RuntimeError: If no GPU available and ALLOW_CPU_INFERENCE is not set.
        """
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

        # CPU fallback - only allowed if explicitly enabled (prevents silent 60x slowdown)
        if os.environ.get("ALLOW_CPU_INFERENCE", "").lower() == "true":
            logger.warning("Using CPU device (ALLOW_CPU_INFERENCE=true)")
            return "cpu"

        raise RuntimeError(
            "No GPU available (CUDA/MPS) and ALLOW_CPU_INFERENCE is not set. "
            "Set ALLOW_CPU_INFERENCE=true to allow CPU inference (very slow). "
            "On cloud, check CUDA driver compatibility with PyTorch base image."
        )

    def _load_predictor(self) -> None:
        """Lazy load SAM3 video model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            # Apply cv_utils fallback if the kernel isn't installed
            # The patch function checks if the kernel loads and only patches if needed
            from .cv_utils_fallback import patch_sam3_cv_utils
            patch_sam3_cv_utils()

            self._device = self._select_device()

            # Determine dtype: bfloat16 is NOT supported on MPS, use float32 instead
            if self._device == "mps":
                if self._config.use_half_precision:
                    logger.warning(
                        "bfloat16 is not supported on MPS (Apple Silicon). "
                        "Using float32 for stable tracking."
                    )
                dtype = torch.float32
            else:
                dtype = torch.bfloat16 if self._config.use_half_precision else torch.float32

            self._dtype = dtype

            # Try local model path first (for Docker), then HuggingFace
            model_path = self._find_model_path()
            logger.info(f"Loading SAM3 from: {model_path}")

            self._model = Sam3VideoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=(model_path != "facebook/sam3"),
            ).to(self._device)

            # Apply torch.compile for GPU speedup
            # Note: The default inductor backend has bugs with SAM3's permute operations,
            # so we use cudagraphs backend which provides GPU optimization without inductor
            if self._config.use_torch_compile and self._device == "cuda":
                logger.info("Applying torch.compile() optimization (this may take a moment)...")
                try:
                    # cudagraphs backend: Uses CUDA graphs for optimization
                    # without triggering inductor's buggy AOT autograd tracing
                    self._model = torch.compile(
                        self._model,
                        backend="cudagraphs",
                        dynamic=False,  # cudagraphs requires static shapes
                    )
                    logger.info("torch.compile() with cudagraphs backend applied successfully")
                except Exception as e:
                    logger.warning(
                        f"torch.compile() failed at setup, running without compilation: {e}"
                    )

            self._processor = Sam3VideoProcessor.from_pretrained(
                model_path,
                local_files_only=(model_path != "facebook/sam3"),
            )

            logger.info("SAM3 video model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "SAM3 not installed. Install with: "
                "uv pip install transformers>=4.48"
            ) from e

    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 1,
        max_frames: int | None = None,
        start_frame: int = 0,
    ) -> Generator[FrameDetections, None, None]:
        """Process video and yield FrameDetections for each frame.

        Uses streaming mode to process one frame at a time. Memory usage
        depends on the `memory_window_size` config setting:

        - window_size=0 (default): Keeps all frame outputs in memory. Provides
          stable tracking IDs but memory grows with video length.

        - window_size>0: Rolling window that prunes frame outputs older than
          this many frames to bound memory. Useful for long videos.

        Args:
            video_path: Path to input video file.
            sample_interval: Process every Nth frame. Default 1 (every frame)
                is strongly recommended for stable tracking IDs.
            max_frames: Maximum number of frames to process (None = all).
            start_frame: Start processing from this frame number.

        Yields:
            FrameDetections for each processed frame.
        """
        import gc

        from PIL import Image

        self._load_predictor()

        video_id = video_path.stem
        window_size = self._config.memory_window_size

        with self._frame_extractor.temp_frame_folder(video_id) as frames_dir:
            # Extract frames to JPEG folder
            extraction = self._frame_extractor.extract_frames(
                video_path,
                frames_dir,
                sample_interval=sample_interval,
                max_frames=max_frames,
                start_frame=start_frame,
            )

            if window_size > 0:
                logger.info(
                    f"Rolling memory window enabled (size={window_size}). "
                    "Frame outputs older than this will be pruned to bound memory."
                )

            logger.info(
                f"Processing {extraction.frame_count} frames with SAM3 streaming mode "
                f"(prompt='{self._config.prompt}', memory_window={window_size})"
            )

            frame_paths = sorted(frames_dir.glob("*.jpg"))
            if not frame_paths:
                logger.warning("No frames extracted")
                return

            # Initialize streaming video session (no video parameter = streaming mode)
            # Device placement strategy:
            # - CUDA: Keep everything on GPU for best performance with torch.compile
            # - MPS: Use CPU for state/storage to reduce memory fragmentation on unified memory
            if self._device == "cuda":
                # CUDA: all on GPU for optimal torch.compile performance
                inference_session = self._processor.init_video_session(
                    inference_device=self._device,
                    inference_state_device=self._device,
                    processing_device=self._device,
                    video_storage_device=self._device,
                    dtype=self._dtype,
                )
            else:
                # MPS/CPU: use CPU for state to reduce memory fragmentation
                inference_session = self._processor.init_video_session(
                    inference_device=self._device,
                    inference_state_device="cpu",
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=self._dtype,
                )

            # Add text prompt for detection and tracking
            inference_session = self._processor.add_text_prompt(
                inference_session=inference_session,
                text=self._config.prompt,
            )

            logger.info(f"Added text prompt: '{self._config.prompt}'")

            # Process frames one-by-one in streaming mode
            for frame_idx, frame_path in enumerate(frame_paths):
                # Load single frame (memory efficient - only one frame at a time)
                frame = Image.open(frame_path)
                frame.load()

                # Process frame through processor
                inputs = self._processor(
                    images=frame,
                    device=self._device,
                    return_tensors="pt",
                )

                # Run streaming inference
                model_outputs = self._model(
                    inference_session=inference_session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )

                # Post-process outputs
                processed_outputs = self._processor.postprocess_outputs(
                    inference_session,
                    model_outputs,
                    original_sizes=inputs.original_sizes,
                )

                # Prune old frame outputs to bound memory (rolling window)
                if window_size > 0:
                    self._prune_old_frames(inference_session, frame_idx, window_size)

                # Get original frame number from extraction mapping
                original_frame_number = extraction.frame_indices[frame_idx]

                # Convert to FrameDetections (continuous IDs, no chunking)
                frame_detections = self._convert_to_frame_detections(
                    processed_outputs,
                    frame_number=original_frame_number,
                    frame_width=extraction.width,
                    frame_height=extraction.height,
                )
                yield frame_detections

                # Log progress periodically
                if (frame_idx + 1) % 100 == 0:
                    logger.info(f"Processed {frame_idx + 1}/{len(frame_paths)} frames...")

                # Free memory - let PIL image be garbage collected
                del frame, inputs

                # Periodic garbage collection
                if (frame_idx + 1) % 50 == 0:
                    gc.collect()

            logger.info(f"Streaming processing complete: {len(frame_paths)} frames")

    def _prune_old_frames(
        self,
        inference_session,
        current_frame_idx: int,
        window_size: int,
    ) -> None:
        """Prune old frame outputs to maintain a rolling memory window.

        This removes per-frame tracking outputs older than window_size frames
        to bound memory usage. Pruning only starts once current_frame_idx
        exceeds window_size.

        Prunes the following structures (see bbva-4f8 for research):
        - non_cond_frame_outputs: Per-frame mask/memory features
        - cond_frame_outputs: Conditioning frame outputs
        - vision_features cache: Computed image embeddings
        - processed_frames: Pixel tensors
        - mask_inputs_per_obj/point_inputs_per_obj: Stored prompts
        - tracker scores and frame tracking metadata

        Args:
            inference_session: The SAM3 video inference session.
            current_frame_idx: Current frame index being processed.
            window_size: Number of recent frames to keep.
        """
        import gc

        if current_frame_idx < window_size:
            return  # Not enough frames yet

        cutoff_frame = current_frame_idx - window_size

        # Prune per-object frame outputs (non-conditioning and conditioning)
        if hasattr(inference_session, "output_dict_per_obj"):
            for obj_idx in list(inference_session.output_dict_per_obj.keys()):
                obj_outputs = inference_session.output_dict_per_obj[obj_idx]

                # Prune non-conditioning frame outputs (the bulk of memory)
                if "non_cond_frame_outputs" in obj_outputs:
                    non_cond = obj_outputs["non_cond_frame_outputs"]
                    frames_to_remove = [
                        f for f in non_cond.keys() if f < cutoff_frame
                    ]
                    for f in frames_to_remove:
                        del non_cond[f]

                # NOTE: Do NOT prune cond_frame_outputs - SAM3 needs these for
                # conditioning context. Pruning causes failures when the window
                # catches up to the initial prompt frame.

        # NOTE: Do NOT prune these - causes MPS dtype mismatch errors:
        # - vision_features cache (image embeddings)
        # - cond_frame_outputs (conditioning context)
        # - processed_frames (pixel tensors)
        # - mask_inputs_per_obj / point_inputs_per_obj (prompts)

        # Prune frame-wise tracker scores (metadata only - safe)
        if hasattr(inference_session, "obj_id_to_tracker_score_frame_wise"):
            for obj_id in list(
                inference_session.obj_id_to_tracker_score_frame_wise.keys()
            ):
                scores = inference_session.obj_id_to_tracker_score_frame_wise[obj_id]
                if isinstance(scores, dict):
                    frames_to_remove = [f for f in scores.keys() if f < cutoff_frame]
                    for f in frames_to_remove:
                        del scores[f]

        # Prune frames_tracked_per_obj
        if hasattr(inference_session, "frames_tracked_per_obj"):
            for obj_idx in list(inference_session.frames_tracked_per_obj.keys()):
                tracked = inference_session.frames_tracked_per_obj[obj_idx]
                if isinstance(tracked, set):
                    inference_session.frames_tracked_per_obj[obj_idx] = {
                        f for f in tracked if f >= cutoff_frame
                    }

        # Force garbage collection
        # NOTE: Do NOT call torch.mps.empty_cache() here - it causes dtype
        # mismatch errors in subsequent MPS matrix multiplications
        gc.collect()

    def _convert_to_frame_detections(
        self,
        sam3_output: dict,
        frame_number: int,
        frame_width: int,
        frame_height: int,
    ) -> FrameDetections:
        """Convert SAM3 output to FrameDetections type.

        Args:
            sam3_output: Output dict from SAM3 postprocess_outputs.
            frame_number: Original video frame number.
            frame_width: Video frame width.
            frame_height: Video frame height.

        Returns:
            FrameDetections compatible with existing pipeline.
        """
        detections = []

        # Extract arrays from output
        object_ids = sam3_output.get("object_ids", np.array([]))
        boxes = sam3_output.get("boxes", np.array([]))
        scores = sam3_output.get("scores", np.array([]))

        # Convert to numpy if tensors (handle GPU tensors by moving to CPU first)
        if hasattr(object_ids, "cpu"):
            object_ids = object_ids.cpu().numpy()
        elif hasattr(object_ids, "numpy"):
            object_ids = object_ids.numpy()
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        elif hasattr(boxes, "numpy"):
            boxes = boxes.numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        elif hasattr(scores, "numpy"):
            scores = scores.numpy()

        for i, obj_id in enumerate(object_ids):
            # Skip low confidence detections
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if confidence < self._config.confidence_threshold:
                continue

            # Convert box from xyxy to BoundingBox
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
