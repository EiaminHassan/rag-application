"""Application settings and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


def _clean_env(value: str) -> str:
    """Trim whitespace and optional wrapping quotes from env values."""
    return value.strip().strip("\"'")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration values."""

    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls(
            gemini_api_key=_clean_env(os.getenv("GEMINI_API_KEY", "")),
            gemini_model=_clean_env(os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
            embedding_model_name=_clean_env(os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            )),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
        )

