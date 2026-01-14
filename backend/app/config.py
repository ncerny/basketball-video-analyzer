from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Basketball Video Analyzer API"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    database_url: str = "sqlite+aiosqlite:///./basketball_analyzer.db"

    video_storage_path: str = "./videos"

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

    ml_models_path: str = "./models"
    ml_device: Literal["cpu", "mps", "cuda", "auto"] = "auto"
    detection_backend: Literal["yolo", "rfdetr"] = "rfdetr"
    tracking_backend: Literal["bytetrack", "norfair", "sam2", "sam3"] = "norfair"
    yolo_model_name: str = "yolov8s.pt"
    yolo_confidence_threshold: float = 0.35
    yolo_person_class_id: int = 0

    # GPU-specific performance settings
    yolo_batch_size_cpu: int = 8
    yolo_batch_size_mps: int = 16
    yolo_batch_size_cuda: int = 32

    # Enable performance logging
    enable_inference_timing: bool = False

    # OCR settings
    enable_jersey_ocr: bool = False  # Disabled while improving tracking
    ocr_sample_rate: int = 10
    ocr_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ocr_max_workers: int = 4

    # Batch processing settings
    batch_frames_per_batch: int = 30
    batch_sample_interval: int = 3
    batch_execution_mode: Literal["sequential", "pipeline", "distributed"] = "sequential"

    # Track merging settings
    enable_jersey_merge: bool = True
    min_jersey_confidence: float = 0.6
    min_jersey_readings: int = 2

    # Identity switch detection settings
    enable_identity_switch_detection: bool = True
    identity_switch_window_size_frames: int = 150
    identity_switch_min_readings: int = 3
    identity_switch_threshold: float = 0.7

    # SAM2 tracking settings
    sam2_model_name: Literal[
        "sam2_hiera_tiny",
        "sam2_hiera_small",
        "sam2_hiera_base_plus",
        "sam2_hiera_large",
    ] = "sam2_hiera_tiny"
    sam2_new_object_iou_threshold: float = 0.3  # IOU below this = new object
    sam2_lost_track_frames: int = 0  # 0 = keep tracks for entire video (no cleanup)
    sam2_max_memory_frames: int = 30  # Limit memory bank size
    sam2_auto_download: bool = True  # Auto-download missing checkpoints

    # SAM2 embedding-based re-identification
    sam2_embedding_similarity_threshold: float = 0.35  # Min cosine similarity for re-ID (lowered for better matching)
    sam2_color_tiebreaker_threshold: float = 0.15  # Use color when embedding scores within this
    sam2_reidentification_enabled: bool = True  # Enable embedding-based re-ID

    # SAM3 tracking settings
    sam3_prompt: str = "basketball player"
    sam3_confidence_threshold: float = 0.25
    sam3_use_half_precision: bool = True
    sam3_temp_frames_dir: Path = Path("/tmp/sam3_frames")

    @property
    def models_dir(self) -> Path:
        path = Path(self.ml_models_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
