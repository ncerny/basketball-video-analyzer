"""SAM2 mask extraction only - no tracking logic.

This module provides precise segmentation masks using SAM2 ImagePredictor,
designed to be used with other trackers (like Norfair) that handle the
actual tracking logic.
"""

import logging
from pathlib import Path

import numpy as np

from app.config import settings

from .sam2_tracker import download_sam2_checkpoint
from .types import BoundingBox, Detection, FrameDetections

logger = logging.getLogger(__name__)


class SAM2MaskExtractor:
    """Extract precise segmentation masks using SAM2 ImagePredictor.

    This class is designed to be used alongside a tracking algorithm
    (like Norfair) that handles temporal association. SAM2 is used
    only for what it's good at: generating precise masks from bounding boxes.

    Usage:
        extractor = SAM2MaskExtractor()
        masks = extractor.extract_masks(frame, detections)
        # Pass masks to color extractor for better histograms
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        min_mask_area: int = 100,
    ):
        """Initialize SAM2 mask extractor.

        Args:
            model_name: SAM2 model name. Defaults to settings.sam2_model_name.
            device: Device to run on ('auto', 'cuda', 'mps', 'cpu').
            min_mask_area: Minimum mask area to consider valid.
        """
        self._model_name = model_name or settings.sam2_model_name
        self._device_config = device
        self._min_mask_area = min_mask_area

        # Lazy-loaded components
        self._model = None
        self._predictor = None
        self._device = None

        logger.info(f"SAM2MaskExtractor initialized with model={self._model_name}")

    def _load_model(self) -> None:
        """Lazy load SAM2 image predictor."""
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Map model name to config file (SAM 2.1 configs)
            model_configs = {
                "sam2_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "sam2_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "sam2_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "sam2_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            }

            config_name = model_configs.get(self._model_name)
            if config_name is None:
                raise ValueError(f"Unknown SAM2 model: {self._model_name}")

            checkpoint_path = settings.models_dir / f"{self._model_name}.pt"

            if not checkpoint_path.exists():
                if settings.sam2_auto_download:
                    logger.info("SAM2 checkpoint not found, auto-downloading...")
                    download_sam2_checkpoint(self._model_name, checkpoint_path)
                else:
                    raise FileNotFoundError(
                        f"SAM2 checkpoint not found: {checkpoint_path}"
                    )

            # Handle device selection
            self._device = self._device_config
            if self._device == "auto":
                import torch

                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            logger.info(f"Loading SAM2 model from {checkpoint_path} on {self._device}...")

            self._model = build_sam2(
                config_name,
                str(checkpoint_path),
                device=self._device,
            )
            self._predictor = SAM2ImagePredictor(self._model)

            logger.info(f"SAM2 model loaded successfully on {self._device}")

        except ImportError as e:
            raise ImportError(
                "SAM2 not installed. Install with: pip install sam2"
            ) from e

    def extract_masks(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[np.ndarray | None]:
        """Generate SAM2 masks for all detections.

        Args:
            frame: BGR image (numpy array).
            detections: List of Detection objects with bounding boxes.

        Returns:
            List of binary masks (bool arrays), same order as detections.
            None for detections where mask extraction failed.
        """
        if not detections:
            return []

        self._load_model()

        # Set image for SAM2 predictor (converts BGR to RGB internally)
        frame_rgb = frame[:, :, ::-1].copy()
        self._predictor.set_image(frame_rgb)

        # Prepare batch of bounding boxes
        boxes = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            boxes.append([x1, y1, x2, y2])

        boxes_array = np.array(boxes, dtype=np.float32)

        try:
            # Batch predict masks for all boxes
            masks, scores, _ = self._predictor.predict(
                box=boxes_array,
                multimask_output=False,  # Single best mask per box
            )

            # Extract and validate masks
            result_masks = []
            for i in range(len(detections)):
                if i < len(masks):
                    mask = masks[i].squeeze()
                    if mask.sum() >= self._min_mask_area:
                        result_masks.append(mask.astype(bool))
                    else:
                        logger.debug(
                            f"Mask {i} too small ({mask.sum()} < {self._min_mask_area})"
                        )
                        result_masks.append(None)
                else:
                    result_masks.append(None)

            logger.debug(
                f"Extracted {sum(1 for m in result_masks if m is not None)}/{len(detections)} masks"
            )
            return result_masks

        except Exception as e:
            logger.warning(f"SAM2 batch prediction failed: {e}")
            return [None] * len(detections)

    def extract_masks_for_frame_detections(
        self,
        frame: np.ndarray,
        frame_detections: FrameDetections,
    ) -> list[np.ndarray | None]:
        """Convenience method that takes FrameDetections directly.

        Args:
            frame: BGR image.
            frame_detections: FrameDetections object.

        Returns:
            List of masks in same order as frame_detections.detections.
        """
        return self.extract_masks(frame, frame_detections.detections)
