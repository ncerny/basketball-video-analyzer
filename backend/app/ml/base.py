from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import FrameDetections


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: Any, frame_number: int = 0) -> "FrameDetections":
        pass

    @abstractmethod
    def detect_batch(
        self, frames: list[Any], start_frame_number: int = 0
    ) -> list["FrameDetections"]:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, str | int | float]:
        pass
