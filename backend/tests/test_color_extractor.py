import numpy as np
import pytest

from app.ml.color_extractor import (
    color_similarity,
    extract_combined_colors,
    extract_jersey_color,
    extract_shoe_color,
)


class TestExtractJerseyColor:
    def test_returns_48_bin_histogram(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_jersey_color(frame, 0, 0, 100, 200)

        assert hist.shape == (48,)
        assert hist.dtype == np.float32

    def test_extracts_full_body(self):
        """Extracts color histogram from full body bbox."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:100, :, :] = [0, 255, 0]
        frame[100:, :, :] = [255, 0, 0]

        hist = extract_jersey_color(frame, 0, 0, 100, 200)

        assert np.linalg.norm(hist) > 0

    def test_empty_bbox_returns_zeros(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_jersey_color(frame, 0, 0, 0, 0)

        assert np.all(hist == 0)

    def test_histogram_is_normalized(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_jersey_color(frame, 0, 0, 100, 200)

        norm = np.linalg.norm(hist)
        assert abs(norm - 1.0) < 1e-5 or norm == 0


class TestExtractShoeColor:
    def test_returns_48_bin_histogram(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_shoe_color(frame, 0, 0, 100, 200)

        assert hist.shape == (48,)
        assert hist.dtype == np.float32

    def test_extracts_full_body_same_as_jersey(self):
        """Both extract_shoe_color and extract_jersey_color now use full body."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:160, :, :] = [0, 255, 0]
        frame[160:, :, :] = [255, 0, 0]

        hist_shoe = extract_shoe_color(frame, 0, 0, 100, 200)
        hist_jersey = extract_jersey_color(frame, 0, 0, 100, 200)

        # Now both should return the same full-body histogram
        assert np.allclose(hist_shoe, hist_jersey)

    def test_empty_bbox_returns_zeros(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_shoe_color(frame, 0, 0, 0, 0)

        assert np.all(hist == 0)

    def test_histogram_is_normalized(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_shoe_color(frame, 0, 0, 100, 200)

        norm = np.linalg.norm(hist)
        assert abs(norm - 1.0) < 1e-5 or norm == 0

    def test_different_body_colors_produce_different_histograms(self):
        """Different full-body colors should produce different histograms."""
        frame1 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame1[:, :, :] = [255, 0, 0]  # All red

        frame2 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame2[:, :, :] = [0, 0, 255]  # All blue

        hist1 = extract_shoe_color(frame1, 0, 0, 100, 200)
        hist2 = extract_shoe_color(frame2, 0, 0, 100, 200)

        similarity = color_similarity(hist1, hist2)
        assert similarity < 0.9


class TestExtractCombinedColors:
    def test_returns_both_histograms(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)

        jersey_hist, shoe_hist = extract_combined_colors(frame, 0, 0, 100, 200)

        assert jersey_hist.shape == (48,)
        assert shoe_hist.shape == (48,)

    def test_jersey_and_shoe_are_same_full_body(self):
        """Both now extract full body, so they should be identical."""
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:100, :, :] = [0, 255, 0]
        frame[160:, :, :] = [255, 0, 0]

        jersey_hist, shoe_hist = extract_combined_colors(frame, 0, 0, 100, 200)

        # Both now use full body, so they should be identical
        assert np.allclose(jersey_hist, shoe_hist)


class TestMaskedColorExtraction:
    def test_mask_excludes_background(self):
        """Mask should exclude background pixels from histogram."""
        # Frame: left half red, right half blue
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :50, :] = [0, 0, 255]  # Red (BGR)
        frame[:, 50:, :] = [255, 0, 0]  # Blue (BGR)

        # Mask only includes left half (red region)
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, :50] = True

        # Without mask: includes both red and blue
        hist_no_mask = extract_jersey_color(frame, 0, 0, 100, 100)

        # With mask: only includes red
        hist_masked = extract_jersey_color(frame, 0, 0, 100, 100, mask=mask)

        # Histograms should be different
        assert not np.allclose(hist_no_mask, hist_masked)

    def test_mask_only_foreground_pixels(self):
        """With mask, only foreground pixels contribute to histogram."""
        # Frame: all green background with red player
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, :] = [0, 255, 0]  # Green background (BGR)
        frame[25:75, 25:75, :] = [0, 0, 255]  # Red player center (BGR)

        # Mask only includes player region
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        # Pure red player crop (no background)
        pure_red = np.zeros((50, 50, 3), dtype=np.uint8)
        pure_red[:, :, :] = [0, 0, 255]

        hist_masked = extract_jersey_color(frame, 0, 0, 100, 100, mask=mask)
        hist_pure_red = extract_jersey_color(pure_red, 0, 0, 50, 50)

        # Masked histogram should match pure red histogram
        assert np.allclose(hist_masked, hist_pure_red, atol=0.01)

    def test_empty_mask_returns_zeros(self):
        """Empty mask (no foreground) should return zero histogram."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)  # All False

        hist = extract_jersey_color(frame, 0, 0, 100, 100, mask=mask)

        assert np.all(hist == 0)

    def test_full_mask_same_as_no_mask(self):
        """Full mask (all foreground) should match no mask."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)  # All True

        hist_no_mask = extract_jersey_color(frame, 0, 0, 100, 100)
        hist_full_mask = extract_jersey_color(frame, 0, 0, 100, 100, mask=mask)

        assert np.allclose(hist_no_mask, hist_full_mask)


class TestColorSimilarity:
    def test_identical_histograms_return_1(self):
        hist = np.random.rand(48).astype(np.float32)
        hist = hist / np.linalg.norm(hist)

        similarity = color_similarity(hist, hist)

        assert abs(similarity - 1.0) < 1e-5

    def test_orthogonal_histograms_return_0(self):
        hist1 = np.zeros(48, dtype=np.float32)
        hist1[0] = 1.0
        hist2 = np.zeros(48, dtype=np.float32)
        hist2[1] = 1.0

        similarity = color_similarity(hist1, hist2)

        assert abs(similarity) < 1e-5

    def test_none_input_returns_0(self):
        hist = np.random.rand(48).astype(np.float32)

        assert color_similarity(None, hist) == 0.0
        assert color_similarity(hist, None) == 0.0
        assert color_similarity(None, None) == 0.0

    def test_zero_norm_returns_0(self):
        hist1 = np.zeros(48, dtype=np.float32)
        hist2 = np.random.rand(48).astype(np.float32)

        similarity = color_similarity(hist1, hist2)

        assert similarity == 0.0
