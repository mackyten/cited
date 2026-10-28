from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from cited.cache import DiskCache
from cited.chunking import Chunk
from cited.client import GeminiClient
from cited.config import Settings
from cited.pipeline import Pipeline
from cited.ratelimit import TokenBucket
from cited.store import Index

DIM = 32


def bow_vector(text: str) -> list[float]:
    """Deterministic bag-of-words embedding so retrieval behaves sensibly offline."""
    v = np.zeros(DIM)
    for tok in text.lower().split():
        v[int(hashlib.md5(tok.strip(".,?:").encode()).hexdigest(), 16) % DIM] += 1
    return v.tolist()


class ApiError(Exception):
    def __init__(self, code: int, msg: str = "") -> None:
        super().__init__(msg or f"{code}")
        self.code = code


class FakeBackend:
    def __init__(self, answer: str = "FastAPI uses Pydantic models [1].") -> None:
        self.answer = answer
        self.embed_calls: list[tuple[int, str]] = []
        self.generate_calls = 0
        self.fail_next: list[Exception] = []

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        self.embed_calls.append((len(texts), task_type))
        if self.fail_next:
            raise self.fail_next.pop(0)
        return [bow_vector(t) for t in texts]

    def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        self.generate_calls += 1
        if self.fail_next:
            raise self.fail_next.pop(0)
        return self.answer, 100, 10

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        yield from ("FastAPI uses ", "Pydantic [1].")

    def is_retryable(self, exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return isinstance(code, int) and (code == 429 or code >= 500)


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=tmp_path, rpm=6000, burst=1000, abstain_threshold=0.2, max_attempts=3)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(backend: FakeBackend, settings: Settings) -> GeminiClient:
    bucket = TokenBucket(6000, 1000, clock=lambda: 0.0, sleep=lambda _s: None)
    return GeminiClient(backend, settings, DiskCache(":memory:"), bucket=bucket,
                        sleep=lambda _s: None)


DOCS = {
    "tutorial/body.md": "Request body: declare a Pydantic model as a function parameter to read the request body.",
    "tutorial/cors.md": "CORS: use CORSMiddleware with a list of allowed origins so browsers permit cross origin calls.",
    "deployment/docker.md": "Docker: build a container image with a Dockerfile and run a single Uvicorn process per container.",
}


@pytest.fixture
def index(client: GeminiClient) -> Index:
    chunks = [Chunk(f"id{i}", src, (src.split("/")[-1][:-3].title(),), None, text,
                    f"https://example.test/{src}") for i, (src, text) in enumerate(DOCS.items())]
    vecs = client.embed([c.embed_text for c in chunks], "RETRIEVAL_DOCUMENT")
    return Index(chunks, vecs, {"n_chunks": len(chunks)})


@pytest.fixture
def pipeline(index: Index, client: GeminiClient, settings: Settings) -> Pipeline:
    return Pipeline(index, client, settings)
