"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "Basketball Video Analyzer API"
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./basketball_analyzer.db"

    # Video storage
    video_storage_path: str = "./videos"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
