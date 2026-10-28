import numpy as np
import pytest

from cited.cache import DiskCache
from cited.client import GeminiClient, QuotaExhausted
from cited.config import Settings
from cited.ratelimit import TokenBucket
from conftest import ApiError, FakeBackend


def test_embed_batches_of_100_and_caches_by_content(client, backend) -> None:  # type: ignore[no-untyped-def]
    texts = [f"text {i}" for i in range(250)]
    v = client.embed(texts, "RETRIEVAL_DOCUMENT")
    assert v.shape[0] == 250 and backend.embed_calls == [(100, "RETRIEVAL_DOCUMENT")] * 2 + [(50, "RETRIEVAL_DOCUMENT")]
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    before = client.requests_made
    client.embed(texts, "RETRIEVAL_DOCUMENT")  # unchanged content -> zero API calls
    assert client.requests_made == before
    client.embed(texts + ["new text"], "RETRIEVAL_DOCUMENT")  # only the new chunk is sent
    assert backend.embed_calls[-1] == (1, "RETRIEVAL_DOCUMENT")


def test_embedding_cache_is_per_task_type(client, backend) -> None:  # type: ignore[no-untyped-def]
    client.embed(["q"], "RETRIEVAL_DOCUMENT")
    client.embed(["q"], "RETRIEVAL_QUERY")
    assert [t for _, t in backend.embed_calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_use_cache_false_forces_live_call(client, backend) -> None:  # type: ignore[no-untyped-def]
    client.embed(["q"], "RETRIEVAL_QUERY")
    client.embed(["q"], "RETRIEVAL_QUERY", use_cache=False)
    assert len(backend.embed_calls) == 2


def test_resumable_after_partial_failure(settings) -> None:  # type: ignore[no-untyped-def]
    be = FakeBackend()
    cache = DiskCache(":memory:")
    bucket = TokenBucket(6000, 1000, clock=lambda: 0.0, sleep=lambda s: None)
    c = GeminiClient(be, settings, cache, bucket=bucket, sleep=lambda s: None)
    texts = [f"t{i}" for i in range(150)]
    orig = be.embed
    state = {"n": 0}

    def flaky(batch, task):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] == 2:
            raise ApiError(400, "bad")  # non-retryable: aborts the run after batch 1 is stored
        return orig(batch, task)

    be.embed = flaky  # type: ignore[method-assign]
    with pytest.raises(ApiError):
        c.embed(texts, "RETRIEVAL_DOCUMENT")
    be.embed = orig  # type: ignore[method-assign]
    be.embed_calls.clear()
    c.embed(texts, "RETRIEVAL_DOCUMENT")
    assert be.embed_calls == [(50, "RETRIEVAL_DOCUMENT")]  # first 100 came from the cache


def test_retries_429_then_succeeds(client, backend) -> None:  # type: ignore[no-untyped-def]
    backend.fail_next = [ApiError(429), ApiError(503)]
    r = client.generate("sys", "prompt")
    assert r.text and client.retries == 2 and client.requests_made == 3


def test_does_not_retry_client_errors(client, backend) -> None:  # type: ignore[no-untyped-def]
    backend.fail_next = [ApiError(400)]
    with pytest.raises(ApiError):
        client.generate("sys", "prompt")
    assert client.requests_made == 1


def test_gives_up_after_max_attempts(client, backend) -> None:  # type: ignore[no-untyped-def]
    backend.fail_next = [ApiError(500)] * 5
    with pytest.raises(ApiError):
        client.generate("s", "p")
    assert client.requests_made == 3  # settings.max_attempts


def test_generation_cache_hits_make_no_request(client, backend) -> None:  # type: ignore[no-untyped-def]
    a = client.generate("s", "p")
    b = client.generate("s", "p")
    assert not a.cached and b.cached and b.latency_ms == a.latency_ms
    assert backend.generate_calls == 1 and client.is_generation_cached("s", "p")
    client.generate("s", "p", use_cache=False)
    assert backend.generate_calls == 2


def test_limiter_is_consulted_per_request(settings) -> None:  # type: ignore[no-untyped-def]
    slept: list[float] = []
    t = {"now": 0.0}
    bucket = TokenBucket(60, 1, clock=lambda: t["now"], sleep=lambda s: (slept.append(s), t.__setitem__("now", t["now"] + s)))
    c = GeminiClient(FakeBackend(), settings, DiskCache(":memory:"), bucket=bucket, sleep=lambda s: None)
    c.generate("s", "a"); c.generate("s", "b"); c.generate("s", "c")
    assert slept == [pytest.approx(1.0), pytest.approx(1.0)]


def test_genai_backend_does_not_retry_daily_quota() -> None:
    from cited.client import GenaiBackend

    be = GenaiBackend.__new__(GenaiBackend)  # no network, classification only
    assert be.is_retryable(ApiError(429, "rate limited")) is True
    assert be.is_retryable(ApiError(429, "Quota exceeded: RequestsPerDay")) is False
    assert be.is_retryable(ApiError(503)) is True
    assert be.is_retryable(ApiError(400)) is False
    assert issubclass(QuotaExhausted, RuntimeError)
