"""Central configuration, loaded from environment variables / a .env file."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_chat_model: str = field(default_factory=lambda: _env("GEMINI_CHAT_MODEL", "gemini-3.8-flash"))
    gemini_embed_model: str = field(default_factory=lambda: _env("GEMINI_EMBED_MODEL", "gemini-embedding-001"))

    # Chunking / retrieval knobs
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "200")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
