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

    def test_extracts_upper_half(self):
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

    def test_extracts_bottom_20_percent(self):
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:160, :, :] = [0, 255, 0]
        frame[160:, :, :] = [255, 0, 0]

        hist_shoe = extract_shoe_color(frame, 0, 0, 100, 200)
        hist_jersey = extract_jersey_color(frame, 0, 0, 100, 200)

        assert not np.allclose(hist_shoe, hist_jersey)

    def test_empty_bbox_returns_zeros(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_shoe_color(frame, 0, 0, 0, 0)

        assert np.all(hist == 0)

    def test_histogram_is_normalized(self):
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        hist = extract_shoe_color(frame, 0, 0, 100, 200)

        norm = np.linalg.norm(hist)
        assert abs(norm - 1.0) < 1e-5 or norm == 0

    def test_different_shoe_colors_produce_different_histograms(self):
        frame1 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame1[160:, :, :] = [255, 0, 0]

        frame2 = np.zeros((200, 100, 3), dtype=np.uint8)
        frame2[160:, :, :] = [0, 0, 255]

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

    def test_jersey_and_shoe_are_different(self):
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:100, :, :] = [0, 255, 0]
        frame[160:, :, :] = [255, 0, 0]

        jersey_hist, shoe_hist = extract_combined_colors(frame, 0, 0, 100, 200)

        similarity = color_similarity(jersey_hist, shoe_hist)
        assert similarity < 0.9


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
