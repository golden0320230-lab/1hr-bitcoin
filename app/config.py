"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import (
    DEFAULT_DB_PATH,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NEWS_ARTICLE_LIMIT,
)

AppEnv = Literal["dev", "test", "prod"]
LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


class Settings(BaseSettings):
    """Application settings with validation and sane defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnv = Field(default="dev", validation_alias="APP_ENV")
    log_level: LogLevel = Field(default="INFO", validation_alias="LOG_LEVEL")
    db_path: Path = Field(default=DEFAULT_DB_PATH, validation_alias="DB_PATH")
    model_path: Path = Field(default=DEFAULT_MODEL_PATH, validation_alias="MODEL_PATH")
    news_article_limit: int = Field(
        default=DEFAULT_NEWS_ARTICLE_LIMIT,
        validation_alias="NEWS_ARTICLE_LIMIT",
        ge=1,
        le=100,
    )
    kimiclaw_base_url: str = Field(
        default="https://replace-me.example.com",
        validation_alias="KIMICLAW_BASE_URL",
    )
    kimiclaw_api_key: SecretStr = Field(
        default=SecretStr("replace-me"),
        validation_alias="KIMICLAW_API_KEY",
    )
    kimiclaw_model: str = Field(default="replace-me", validation_alias="KIMICLAW_MODEL")
    http_timeout_seconds: float = Field(
        default=DEFAULT_HTTP_TIMEOUT_SECONDS,
        validation_alias="HTTP_TIMEOUT_SECONDS",
        gt=0,
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str | LogLevel) -> str | LogLevel:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("db_path", "model_path", mode="before")
    @classmethod
    def normalize_paths(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    def ensure_runtime_directories(self) -> None:
        """Create local directories needed by the configured paths."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, *, log_level: str | None = None) -> Settings:
        """Return a validated copy with command-line overrides applied."""

        data = self.model_dump()
        if log_level is not None:
            data["log_level"] = log_level
        return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings for the current process."""

    return Settings()
