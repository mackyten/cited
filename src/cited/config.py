"""Settings and API key loading. The key is never printed or logged."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_APIKEY")

ABSTAIN_SENTENCE = "I don't know based on the documentation."


@dataclass(frozen=True)
class Settings:
    text_model: str = "gemini-3.8-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    data_dir: Path = field(default_factory=lambda: Path("data"))
    corpus_repo: str = "https://github.com/fastapi/fastapi.git"
    corpus_subdir: str = "docs/en/docs"
    docs_base_url: str = "https://fastapi.tiangolo.com"
    max_chunk_chars: int = 1500
    overlap_chars: int = 150
    min_chunk_chars: int = 80
    top_k: int = 5
    max_output_tokens: int = 600
    temperature: float = 0.1
    rpm: float = 20.0  # token-bucket refill, requests per minute
    burst: int = 5
    max_attempts: int = 6
    request_budget: int = 150  # refuse commands estimated above this unless --yes
    abstain_threshold: float = 0.717  # min dense cosine of top hit; midpoint of the gap on eval/dev.jsonl (n=13)

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "fastapi-docs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache.sqlite"


def load_settings() -> Settings:
    """Defaults, with a few CITED_* environment overrides."""
    s = Settings()
    overrides: dict[str, object] = {}
    if v := os.environ.get("CITED_TEXT_MODEL"):
        overrides["text_model"] = v
    if v := os.environ.get("CITED_RPM"):
        overrides["rpm"] = float(v)
    if v := os.environ.get("CITED_DATA_DIR"):
        overrides["data_dir"] = Path(v)
    if v := os.environ.get("CITED_BUDGET"):
        overrides["request_budget"] = int(v)
    if v := os.environ.get("CITED_ABSTAIN_THRESHOLD"):
        overrides["abstain_threshold"] = float(v)
    return Settings(**{**s.__dict__, **overrides}) if overrides else s


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out


def load_api_key(dotenv: Path = Path(".env")) -> str | None:
    """Return the key from the environment, else from a gitignored .env."""
    env_file = _read_dotenv(dotenv)
    for name in KEY_ENV_NAMES:
        val = os.environ.get(name) or env_file.get(name)
        if val:
            return val
    return None


class MissingKeyError(RuntimeError):
    pass


def require_api_key() -> str:
    key = load_api_key()
    if not key:
        raise MissingKeyError(
            "No Gemini API key found. Export GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
            "terminal, or put it in a gitignored .env file. Do not paste it into chat."
        )
    return key
