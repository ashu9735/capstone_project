"""Central configuration. Every tunable value is read from the environment exactly once."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Values people leave in .env after copying the template. Treated as absent.
PLACEHOLDER_KEYS = {"", "your_key_here", "your-key-here", "changeme", "none", "null"}


def is_real_key(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in PLACEHOLDER_KEYS


def _abs(path: str) -> Path:
    """Resolve relative paths against the project root, not the caller's cwd.

    Chroma silently returns nothing when written and read through different
    relative paths, which is the single most common retrieval failure.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # Model access
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "meta-llama/llama-3.1-8b-instruct"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    fallback_model_name: str = "llama-3.1-8b-instant"
    enable_llm: bool = True

    # Embeddings and vector store
    embedding_backend: str = "onnx"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "cloudserve_docs"
    chunk_size: int = 400
    chunk_overlap: int = 60
    retrieval_top_k: int = 5
    retrieval_max_distance: float = 0.55
    retrieval_strategy: str = "dense"
    retrieval_candidate_multiplier: int = 3
    retrieval_rrf_k: int = 60

    # Persistence
    database_url: str = "sqlite:///./storage/decisions.db"

    # Routing
    confidence_threshold: float = 0.80
    use_calibration: bool = True
    allowed_response_domains: str = ""

    # Runtime
    log_level: str = "INFO"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    ticket_timeout_seconds: float = 90.0
    max_concurrency: int = 4
    random_seed: int = 42

    # Data
    docs_path: str = "./data/documentation.json"

    # Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    metrics_port: int = 8001
    kill_switch: bool = False

    system_version: str = Field(default="1.0.0")

    @property
    def chroma_dir(self) -> Path:
        return _abs(self.chroma_path)

    @property
    def documentation_file(self) -> Path:
        return _abs(self.docs_path)

    @property
    def sqlite_file(self) -> Path:
        url = self.database_url
        if not url.startswith("sqlite"):
            raise ValueError(f"Only sqlite is supported, got: {url}")
        return _abs(url.split("///", 1)[1])

    @property
    def llm_available(self) -> bool:
        return self.enable_llm and bool(self.configured_providers)

    @property
    def configured_providers(self) -> list[str]:
        names = []
        if is_real_key(self.openrouter_api_key):
            names.append("openrouter")
        if is_real_key(self.groq_api_key):
            names.append("groq")
        return names


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return get_settings()


__all__ = ["Settings", "get_settings", "reload_settings", "PROJECT_ROOT", "is_real_key", "os"]
