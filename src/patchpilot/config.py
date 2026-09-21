from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for PatchPilot."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GITHUB_TOKEN: str = ""
    DEFAULT_OPUS_MODEL: str = "claude-sonnet-4-20250514"
    DEFAULT_GEMINI_MODEL: str = "gemini-2.0-flash"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RETRIES: int = 3
    SANDBOX_TIMEOUT: int = 300
    SANDBOX_MODE: Literal["docker", "subprocess"] = "subprocess"
    LOG_LEVEL: str = "INFO"
    TRACE_DIR: str = "./traces"


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached instance of the configuration."""
    return Config()
