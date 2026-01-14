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
