from dataclasses import dataclass
import logging
import re
import threading
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    raw_text: str
    parsed_number: int | None
    confidence: float
    is_valid: bool


@dataclass
class OCRConfig:
    model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    max_new_tokens: int = 32
    device: str = "auto"


JERSEY_NUMBER_PROMPT = "What is the jersey number visible in this image? Reply with ONLY the number, nothing else. If no number is visible, reply 'none'."

# Minimum crop dimensions for OCR - SmolVLM2 uses 32x32 patches
# Crops smaller than this can't produce meaningful patches and may cause reshape errors
MIN_CROP_WIDTH = 32
MIN_CROP_HEIGHT = 32


class JerseyOCR:
    _load_lock = threading.Lock()

    def __init__(self, config: OCRConfig | None = None) -> None:
        self._config = config or OCRConfig()
        self._model: "AutoModelForImageTextToText | None" = None
        self._processor: "AutoProcessor | None" = None
        self._device: str = "cpu"
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch

            self._processor = AutoProcessor.from_pretrained(self._config.model_name)

            self._device = self._config.device
            if self._config.device == "auto":
                if torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

            use_dtype = torch.float16 if self._device != "cpu" else torch.float32

            self._model = AutoModelForImageTextToText.from_pretrained(
                self._config.model_name,
                torch_dtype=use_dtype,
                device_map=self._device if self._device not in ("mps", "cpu") else None,
            )

            if self._device in ("mps", "cpu"):
                self._model = self._model.to(self._device)

            self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def read_jersey_number(self, crop: np.ndarray) -> OCRResult:
        self._load_model()

        if crop.size == 0:
            return OCRResult(raw_text="", parsed_number=None, confidence=0.0, is_valid=False)

        # Validate minimum dimensions - very small crops cause VLM reshape errors
        if len(crop.shape) >= 2:
            height, width = crop.shape[:2]
            if height < MIN_CROP_HEIGHT or width < MIN_CROP_WIDTH:
                return OCRResult(
                    raw_text="", parsed_number=None, confidence=0.0, is_valid=False
                )

        if len(crop.shape) == 3 and crop.shape[2] == 3:
            image = Image.fromarray(crop[:, :, ::-1])
        else:
            image = Image.fromarray(crop)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": JERSEY_NUMBER_PROMPT},
                    ],
                }
            ]

            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self._processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
            )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            raw_text = self._processor.decode(generated_ids, skip_special_tokens=True)
            raw_text = raw_text.strip()

        except Exception as e:
            # MPS and concurrent VLM inference can cause shape mismatches
            # Log and return invalid result instead of crashing the pipeline
            logger.warning(
                f"OCR inference failed for crop {crop.shape}: {type(e).__name__}: {e}"
            )
            return OCRResult(
                raw_text="", parsed_number=None, confidence=0.0, is_valid=False
            )

        parsed_number, confidence, is_valid = self._parse_jersey_number(raw_text)

        return OCRResult(
            raw_text=raw_text,
            parsed_number=parsed_number,
            confidence=confidence,
            is_valid=is_valid,
        )

    def _parse_jersey_number(self, raw_text: str) -> tuple[int | None, float, bool]:
        text = raw_text.lower().strip()

        if text in ("none", "no number", "not visible", "n/a", ""):
            return None, 0.5, False

        numbers = re.findall(r"\d{1,2}", text)
        if not numbers:
            return None, 0.3, False

        number_str = numbers[0]
        try:
            number = int(number_str)
        except ValueError:
            return None, 0.2, False

        if number < 0 or number > 99:
            return None, 0.4, False

        confidence = 0.9 if len(numbers) == 1 else 0.7

        return number, confidence, True


_global_ocr: JerseyOCR | None = None

# Thread-local storage for parallel OCR - each thread gets its own model instance
# This avoids thread-safety issues with PyTorch models during concurrent inference
_thread_local = threading.local()


def get_jersey_ocr(config: OCRConfig | None = None) -> JerseyOCR:
    """Get a global OCR instance (for single-threaded use only)."""
    global _global_ocr
    if _global_ocr is None:
        _global_ocr = JerseyOCR(config)
    return _global_ocr


def get_thread_local_ocr(config: OCRConfig | None = None) -> JerseyOCR:
    """Get a thread-local OCR instance for safe parallel inference.

    Each thread gets its own JerseyOCR instance with its own model copy.
    This allows true parallel OCR processing without thread-safety issues.

    Args:
        config: OCR configuration. Only used when creating a new instance.

    Returns:
        Thread-local JerseyOCR instance.
    """
    if not hasattr(_thread_local, "ocr") or _thread_local.ocr is None:
        _thread_local.ocr = JerseyOCR(config)
    return _thread_local.ocr


def read_jersey_number(crop: np.ndarray, config: OCRConfig | None = None) -> OCRResult:
    ocr = get_jersey_ocr(config)
    return ocr.read_jersey_number(crop)


def read_jersey_number_thread_safe(
    crop: np.ndarray, config: OCRConfig | None = None
) -> OCRResult:
    """Thread-safe jersey number reading using thread-local model instance.

    Use this function when calling from multiple threads (e.g., ThreadPoolExecutor).

    Args:
        crop: Image crop containing jersey.
        config: OCR configuration.

    Returns:
        OCR result with parsed jersey number.
    """
    ocr = get_thread_local_ocr(config)
    return ocr.read_jersey_number(crop)
