from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.jersey_ocr import JerseyOCR, OCRConfig, OCRResult


class TestOCRResult:
    def test_valid_result(self) -> None:
        result = OCRResult(raw_text="23", parsed_number=23, confidence=0.9, is_valid=True)
        assert result.is_valid
        assert result.parsed_number == 23

    def test_invalid_result(self) -> None:
        result = OCRResult(raw_text="none", parsed_number=None, confidence=0.5, is_valid=False)
        assert not result.is_valid
        assert result.parsed_number is None


class TestOCRConfig:
    def test_default_config(self) -> None:
        config = OCRConfig()
        assert "SmolVLM2" in config.model_name
        assert config.device == "auto"
        assert config.max_new_tokens == 32

    def test_custom_config(self) -> None:
        config = OCRConfig(model_name="custom/model", device="cpu", max_new_tokens=64)
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.max_new_tokens == 64


class TestJerseyOCR:
    def test_init_not_loaded(self) -> None:
        ocr = JerseyOCR()
        assert not ocr.is_loaded()

    def test_parse_jersey_number_valid_single(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("23")
        assert number == 23
        assert confidence == 0.9
        assert is_valid

    def test_parse_jersey_number_valid_with_text(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("The number is 42")
        assert number == 42
        assert confidence == 0.9
        assert is_valid

    def test_parse_jersey_number_multiple_numbers(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("I see 23 and 45")
        assert number == 23
        assert confidence == 0.7
        assert is_valid

    def test_parse_jersey_number_none_response(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("none")
        assert number is None
        assert confidence == 0.5
        assert not is_valid

    def test_parse_jersey_number_no_number(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("no visible number")
        assert number is None
        assert confidence == 0.3
        assert not is_valid

    def test_parse_jersey_number_three_digits(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("100")
        assert number == 10
        assert confidence == 0.7
        assert is_valid

    def test_parse_jersey_number_large_garbage(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("12345678901234567890")
        assert number == 12
        assert confidence == 0.7
        assert is_valid

    def test_parse_jersey_number_zero(self) -> None:
        ocr = JerseyOCR()
        number, confidence, is_valid = ocr._parse_jersey_number("0")
        assert number == 0
        assert confidence == 0.9
        assert is_valid

    def test_read_empty_crop(self) -> None:
        ocr = JerseyOCR()
        ocr._loaded = True
        empty_crop = np.array([], dtype=np.uint8)
        result = ocr.read_jersey_number(empty_crop)
        assert not result.is_valid
        assert result.parsed_number is None

    def test_read_small_crop_rejected(self) -> None:
        """Very small crops are rejected to avoid VLM reshape errors."""
        ocr = JerseyOCR()
        ocr._loaded = True
        # Crop smaller than MIN_CROP_WIDTH x MIN_CROP_HEIGHT (32x32)
        small_crop = np.zeros((20, 20, 3), dtype=np.uint8)
        result = ocr.read_jersey_number(small_crop)
        assert not result.is_valid
        assert result.parsed_number is None
        assert result.confidence == 0.0

    def test_read_narrow_crop_rejected(self) -> None:
        """Crops with one dimension too small are rejected."""
        ocr = JerseyOCR()
        ocr._loaded = True
        # Wide but too short
        narrow_crop = np.zeros((20, 100, 3), dtype=np.uint8)
        result = ocr.read_jersey_number(narrow_crop)
        assert not result.is_valid

    @patch("app.ml.jersey_ocr.AutoModelForImageTextToText")
    @patch("app.ml.jersey_ocr.AutoProcessor")
    def test_load_model_cpu(self, mock_processor_cls: MagicMock, mock_model_cls: MagicMock) -> None:
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        config = OCRConfig(device="cpu")
        ocr = JerseyOCR(config)
        ocr._load_model()

        assert ocr.is_loaded()
        assert ocr._device == "cpu"
        mock_model.to.assert_called_once_with("cpu")

    @patch("app.ml.jersey_ocr.AutoModelForImageTextToText")
    @patch("app.ml.jersey_ocr.AutoProcessor")
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_load_model_auto_mps(
        self,
        mock_mps: MagicMock,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        config = OCRConfig(device="auto")
        ocr = JerseyOCR(config)
        ocr._load_model()

        assert ocr._device == "mps"

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    def test_load_model_cpu(self, mock_processor_cls: MagicMock, mock_model_cls: MagicMock) -> None:
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        config = OCRConfig(device="cpu")
        ocr = JerseyOCR(config)
        ocr._load_model()

        assert ocr.is_loaded()
        assert ocr._device == "cpu"
        mock_model.to.assert_called_once_with("cpu")

    @patch("transformers.AutoModelForImageTextToText")
    @patch("transformers.AutoProcessor")
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_load_model_auto_mps(
        self,
        mock_mps: MagicMock,
        mock_processor_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        config = OCRConfig(device="auto")
        ocr = JerseyOCR(config)
        ocr._load_model()

        assert ocr._device == "mps"

    def test_read_jersey_number_with_mock_model(self) -> None:
        ocr = JerseyOCR()

        ocr._loaded = True
        ocr._device = "cpu"
        ocr._processor = MagicMock()
        ocr._model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 10]))
        mock_inputs.to.return_value = mock_inputs
        ocr._processor.return_value = mock_inputs
        ocr._processor.apply_chat_template.return_value = "prompt"
        ocr._processor.decode.return_value = "23"
        ocr._model.generate.return_value = [[0] * 15]

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ocr.read_jersey_number(crop)

        assert result.parsed_number == 23
        assert result.is_valid
        assert result.raw_text == "23"

    def test_read_jersey_number_bgr_to_rgb_conversion(self) -> None:
        ocr = JerseyOCR()

        ocr._loaded = True
        ocr._device = "cpu"
        ocr._processor = MagicMock()
        ocr._model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 10]))
        mock_inputs.to.return_value = mock_inputs
        ocr._processor.return_value = mock_inputs
        ocr._processor.apply_chat_template.return_value = "prompt"
        ocr._processor.decode.return_value = "45"
        ocr._model.generate.return_value = [[0] * 15]

        bgr_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_crop[:, :, 0] = 255
        result = ocr.read_jersey_number(bgr_crop)

        assert result.parsed_number == 45
        assert result.is_valid

    def test_inference_error_returns_invalid_result(self) -> None:
        """Model errors during inference return invalid result instead of raising."""
        ocr = JerseyOCR()

        ocr._loaded = True
        ocr._device = "cpu"
        ocr._processor = MagicMock()
        ocr._model = MagicMock()

        # Simulate an error during inference
        ocr._processor.apply_chat_template.side_effect = RuntimeError(
            "shape mismatch: indexing tensors could not be broadcast"
        )

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ocr.read_jersey_number(crop)

        # Should return invalid result instead of raising
        assert not result.is_valid
        assert result.parsed_number is None
        assert result.confidence == 0.0
