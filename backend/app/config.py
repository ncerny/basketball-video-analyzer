from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Basketball Video Analyzer API"
    debug: bool = False

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
    yolo_model_name: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_person_class_id: int = 0

    @property
    def models_dir(self) -> Path:
        path = Path(self.ml_models_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
