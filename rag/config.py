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
    # Which provider to use for embeddings + generation: "gemini" or "openai"
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini").lower())

    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_chat_model: str = field(default_factory=lambda: _env("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))
    gemini_embed_model: str = field(default_factory=lambda: _env("GEMINI_EMBED_MODEL", "gemini-embedding-001"))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embed_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    # Chunking / retrieval knobs
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "200")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    @property
    def api_key(self) -> str:
        return self.gemini_api_key if self.provider == "gemini" else self.openai_api_key

    def validate(self) -> None:
        if self.provider not in {"gemini", "openai"}:
            raise ValueError(f"LLM_PROVIDER must be 'gemini' or 'openai', got '{self.provider}'.")
        if not self.api_key:
            key_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
            raise ValueError(f"{key_name} is not set. Copy .env.example to .env and add your key.")
