"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader so python-dotenv is not a hard dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    corpus_dir: Path = Path(os.getenv("CORPUS_DIR", ROOT / "data" / "corpus"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma"))
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", ROOT / "data" / "zoning_rag.db"))
    collection: str = os.getenv("CHROMA_COLLECTION", "nyc_zoning")

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", 700))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", 5))
    rrf_k: int = int(os.getenv("RRF_K", 60))
    bm25_weight: float = _f("BM25_WEIGHT", 1.0)
    dense_weight: float = _f("DENSE_WEIGHT", 1.0)

    # Grounding / abstention
    min_retrieval_confidence: float = _f("MIN_RETRIEVAL_CONFIDENCE", 0.40)
    min_sentence_support: float = _f("MIN_SENTENCE_SUPPORT", 0.50)
    min_grounded_ratio: float = _f("MIN_GROUNDED_RATIO", 0.60)

    # LLM providers, tried in order; "extractive" is the offline fallback
    llm_providers: list[str] = field(
        default_factory=lambda: os.getenv("LLM_PROVIDERS", "groq,gemini,extractive").split(",")
    )
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm_timeout: float = _f("LLM_TIMEOUT", 30)

    # Embeddings: "auto" tries sentence-transformers, then falls back to hashing
    embedder: str = os.getenv("EMBEDDER", "auto")
    st_model: str = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
