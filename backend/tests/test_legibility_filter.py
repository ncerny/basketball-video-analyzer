import numpy as np
import pytest

from app.ml.legibility_filter import (
    LegibilityConfig,
    check_legibility,
    compute_blur_score,
    extract_jersey_crop,
    filter_legible_detections,
)
from app.ml.types import BoundingBox


class TestComputeBlurScore:
    def test_empty_crop_returns_zero(self) -> None:
        empty_crop = np.array([], dtype=np.uint8)
        assert compute_blur_score(empty_crop) == 0.0

    def test_uniform_image_is_blurry(self) -> None:
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        score = compute_blur_score(uniform)
        assert score < 1.0

    def test_high_contrast_image_is_sharp(self) -> None:
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[::2, :, :] = 255
        score = compute_blur_score(sharp)
        assert score > 100.0


class TestExtractJerseyCrop:
    def test_extracts_upper_half_by_default(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:100, :, :] = 255
        bbox = BoundingBox(x=50, y=0, width=100, height=200)

        crop = extract_jersey_crop(frame, bbox)

        assert crop.shape == (100, 100, 3)
        assert np.all(crop == 255)

    def test_custom_jersey_fraction(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=0, y=0, width=100, height=100)

        crop = extract_jersey_crop(frame, bbox, jersey_fraction=0.25)

        assert crop.shape == (25, 100, 3)

    def test_bbox_outside_frame_clipped(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=-50, y=-50, width=200, height=200)

        crop = extract_jersey_crop(frame, bbox)

        assert crop.shape[0] > 0
        assert crop.shape[1] == 100

    def test_invalid_bbox_returns_empty(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=200, y=200, width=50, height=50)

        crop = extract_jersey_crop(frame, bbox)

        assert crop.size == 0


class TestCheckLegibility:
    @pytest.fixture
    def sharp_frame(self) -> np.ndarray:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[::2, :, :] = 255
        return frame

    @pytest.fixture
    def blurry_frame(self) -> np.ndarray:
        return np.ones((500, 500, 3), dtype=np.uint8) * 128

    @pytest.fixture
    def good_bbox(self) -> BoundingBox:
        return BoundingBox(x=100, y=100, width=100, height=200)

    def test_all_checks_pass(self, sharp_frame: np.ndarray, good_bbox: BoundingBox) -> None:
        result = check_legibility(sharp_frame, good_bbox, confidence=0.9)

        assert result.is_legible is True
        assert result.score > 0.5
        assert len(result.reasons) == 0

    def test_bbox_too_small_fails(self, sharp_frame: np.ndarray) -> None:
        small_bbox = BoundingBox(x=100, y=100, width=20, height=40)

        result = check_legibility(sharp_frame, small_bbox, confidence=0.9)

        assert result.is_legible is False
        assert any("bbox_too_small" in r for r in result.reasons)

    def test_low_confidence_fails(self, sharp_frame: np.ndarray, good_bbox: BoundingBox) -> None:
        result = check_legibility(sharp_frame, good_bbox, confidence=0.5)

        assert result.is_legible is False
        assert any("low_confidence" in r for r in result.reasons)

    def test_too_blurry_fails(self, blurry_frame: np.ndarray, good_bbox: BoundingBox) -> None:
        result = check_legibility(blurry_frame, good_bbox, confidence=0.9)

        assert result.is_legible is False
        assert any("too_blurry" in r for r in result.reasons)

    def test_aspect_ratio_too_low_fails(self, sharp_frame: np.ndarray) -> None:
        wide_bbox = BoundingBox(x=100, y=100, width=200, height=100)

        result = check_legibility(sharp_frame, wide_bbox, confidence=0.9)

        assert result.is_legible is False
        assert any("aspect_ratio_too_low" in r for r in result.reasons)

    def test_aspect_ratio_too_high_fails(self, sharp_frame: np.ndarray) -> None:
        tall_bbox = BoundingBox(x=100, y=100, width=20, height=200)
        config = LegibilityConfig(min_bbox_area=500)

        result = check_legibility(sharp_frame, tall_bbox, confidence=0.9, config=config)

        assert result.is_legible is False
        assert any("aspect_ratio_too_high" in r for r in result.reasons)

    def test_custom_config_thresholds(self, good_bbox: BoundingBox) -> None:
        noisy_frame = np.random.randint(100, 150, (500, 500, 3), dtype=np.uint8)

        lenient_config = LegibilityConfig(
            min_bbox_area=100,
            min_confidence=0.3,
            min_blur_variance=1.0,
            min_aspect_ratio=0.5,
        )

        result = check_legibility(noisy_frame, good_bbox, confidence=0.5, config=lenient_config)

        assert result.is_legible is True

    def test_empty_crop_fails(self, sharp_frame: np.ndarray) -> None:
        outside_bbox = BoundingBox(x=1000, y=1000, width=100, height=200)

        result = check_legibility(sharp_frame, outside_bbox, confidence=0.9)

        assert result.is_legible is False
        assert any("empty_crop" in r for r in result.reasons)


class TestFilterLegibleDetections:
    def test_filters_illegible_detections(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[::2, :, :] = 255

        bboxes = [
            BoundingBox(x=100, y=100, width=100, height=200),
            BoundingBox(x=100, y=100, width=10, height=20),
            BoundingBox(x=100, y=100, width=80, height=160),
        ]
        confidences = [0.9, 0.9, 0.9]

        results = filter_legible_detections(frame, bboxes, confidences)

        passed_indices = [r[0] for r in results]
        assert 0 in passed_indices
        assert 1 not in passed_indices
        assert 2 in passed_indices

    def test_returns_empty_for_all_illegible(self) -> None:
        blurry_frame = np.ones((500, 500, 3), dtype=np.uint8) * 128
        bboxes = [BoundingBox(x=100, y=100, width=100, height=200)]
        confidences = [0.9]

        results = filter_legible_detections(blurry_frame, bboxes, confidences)

        assert len(results) == 0

    def test_includes_legibility_result(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[::2, :, :] = 255
        bboxes = [BoundingBox(x=100, y=100, width=100, height=200)]
        confidences = [0.9]

        results = filter_legible_detections(frame, bboxes, confidences)

        assert len(results) == 1
        idx, legibility_result = results[0]
        assert idx == 0
        assert legibility_result.is_legible is True
        assert legibility_result.score > 0
