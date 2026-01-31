"""Application configuration using pydantic-settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Redis state storage (separate DB for app state like cancellation flags)
    REDIS_STATE_DB: str = "redis://redis:6379/1"

    # Job storage path
    JOB_STORAGE_PATH: str = "/app/storage/jobs"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
