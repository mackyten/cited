"""All Gemini traffic goes through here: rate limit, backoff, caching, request counting."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cited.cache import DiskCache, digest
from cited.config import Settings
from cited.ratelimit import TokenBucket, retry_call

EMBED_BATCH = 100  # hard API limit observed: "at most 100 requests can be in one batch"


class QuotaExhausted(RuntimeError):
    """A per-day (or similar non-recoverable) quota is used up. Retrying will not help."""


class Backend(Protocol):
    """The raw API surface. The real one wraps google-genai, tests inject a fake."""

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]: ...
    def generate(self, system: str, prompt: str) -> tuple[str, int, int]: ...
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...
    def is_retryable(self, exc: Exception) -> bool: ...


class GenaiBackend:
    def __init__(self, api_key: str, settings: Settings) -> None:
        from google import genai  # imported lazily so offline tests never need it

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._s = settings

    def _config(self, system: str):  # type: ignore[no-untyped-def]
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system, temperature=self._s.temperature,
            max_output_tokens=self._s.max_output_tokens,
            # Thinking tokens would otherwise eat the output cap (observed: 57 for one word).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        r = self._client.models.embed_content(
            model=self._s.embed_model, contents=list(texts),
            config=types.EmbedContentConfig(task_type=task_type,
                                            output_dimensionality=self._s.embed_dim))
        return [list(e.values) for e in (r.embeddings or [])]

    def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        r = self._client.models.generate_content(
            model=self._s.text_model, contents=prompt, config=self._config(system))
        u = r.usage_metadata
        return (r.text or "", (u.prompt_token_count or 0) if u else 0,
                (u.candidates_token_count or 0) if u else 0)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        for part in self._client.models.generate_content_stream(
                model=self._s.text_model, contents=prompt, config=self._config(system)):
            if part.text:
                yield part.text

    def is_retryable(self, exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            if code == 429:
                msg = str(exc).lower()
                return not ("perday" in msg or "per day" in msg)  # daily cap: do not hammer
            return code >= 500
        return isinstance(exc, (ConnectionError, TimeoutError))


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float  # latency of the original (uncached) API call
    cached: bool


class GeminiClient:
    def __init__(self, backend: Backend, settings: Settings, cache: DiskCache | None = None,
                 *, bucket: TokenBucket | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None) -> None:
        self.backend, self.s, self.cache = backend, settings, cache
        self.bucket = bucket or TokenBucket(settings.rpm, settings.burst)
        self._sleep = sleep
        self._log = log or (lambda _m: None)
        self.requests_made = 0  # real API requests only (cache hits excluded)
        self.retries = 0  # retried attempts (429/5xx)
        self.waited_s = 0.0  # total time spent blocked in our own rate limiter

    # -- internals ---------------------------------------------------------------------
    def _call(self, fn: Callable[[], object]):  # type: ignore[no-untyped-def]
        def attempt():  # type: ignore[no-untyped-def]
            self.waited_s += self.bucket.acquire()
            self.requests_made += 1
            return fn()

        def note(n: int, e: Exception, d: float) -> None:
            self.retries += 1
            self._log(f"retry {n + 1} in {d:.1f}s after {type(e).__name__}")

        return retry_call(attempt, is_retryable=self.backend.is_retryable,
                          max_attempts=self.s.max_attempts, sleep=self._sleep, on_retry=note)

    def _emb_key(self, text: str, task: str) -> str:
        return "emb:" + digest(self.s.embed_model, str(self.s.embed_dim), task, text)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n == 0, 1, n)

    # -- embeddings --------------------------------------------------------------------
    def count_uncached_embeddings(self, texts: Sequence[str], task: str) -> int:
        if not self.cache:
            return len(set(texts))
        return len({t for t in texts if self._emb_key(t, task) not in self.cache})

    def embed(self, texts: Sequence[str], task: str, *, use_cache: bool = True) -> np.ndarray:
        """Unit-normalised float32 vectors. Cached per text, written after every batch (resumable).

        use_cache=False skips cache reads (results are still stored); used to time live queries.
        """
        out: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for t in dict.fromkeys(texts):
            raw = self.cache.get(self._emb_key(t, task)) if (self.cache and use_cache) else None
            if raw is not None:
                out[t] = np.frombuffer(raw, dtype=np.float32)
            else:
                missing.append(t)
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i:i + EMBED_BATCH]
            vecs = self._normalize(np.asarray(
                self._call(lambda b=batch: self.backend.embed(b, task)), dtype=np.float32))
            for t, v in zip(batch, vecs, strict=True):
                out[t] = v
                if self.cache:
                    self.cache.set(self._emb_key(t, task), v.tobytes())
        return np.stack([out[t] for t in texts]) if texts else np.zeros((0, self.s.embed_dim),
                                                                        dtype=np.float32)

    # -- generation --------------------------------------------------------------------
    def _gen_key(self, system: str, prompt: str) -> str:
        return "gen:" + digest(self.s.text_model, str(self.s.temperature),
                               str(self.s.max_output_tokens), system, prompt)

    def is_generation_cached(self, system: str, prompt: str) -> bool:
        return bool(self.cache) and self._gen_key(system, prompt) in self.cache  # type: ignore[operator]

    def generate(self, system: str, prompt: str, *, use_cache: bool = True) -> GenResult:
        key = self._gen_key(system, prompt)
        if use_cache and self.cache and (raw := self.cache.get(key)) is not None:
            d = json.loads(raw)
            return GenResult(d["text"], d["pt"], d["ot"], d["ms"], cached=True)
        t0, w0 = time.perf_counter(), self.waited_s
        text, pt, ot = self._call(lambda: self.backend.generate(system, prompt))  # type: ignore[misc]
        # Latency excludes time spent waiting in our own limiter (not API latency).
        ms = (time.perf_counter() - t0 - (self.waited_s - w0)) * 1000
        if self.cache:
            self.cache.set(key, json.dumps({"text": text, "pt": pt, "ot": ot, "ms": ms}).encode())
        return GenResult(text, pt, ot, ms, cached=False)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Streaming is never cached. Rate-limited and counted as one request."""
        self.waited_s += self.bucket.acquire()
        self.requests_made += 1
        yield from self.backend.stream(system, prompt)
