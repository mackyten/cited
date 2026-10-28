import pytest

from cited.ratelimit import TokenBucket, backoff_delay, retry_call


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_bucket_allows_burst_then_waits() -> None:
    c = Clock()
    b = TokenBucket(rate_per_min=60, capacity=2, clock=c.now, sleep=c.sleep)  # 1 token / s
    assert b.acquire() == 0 and b.acquire() == 0
    assert b.acquire() == pytest.approx(1.0)
    assert c.slept == [pytest.approx(1.0)]


def test_bucket_refills_over_time_and_caps() -> None:
    c = Clock()
    b = TokenBucket(60, 2, clock=c.now, sleep=c.sleep)
    b.acquire(); b.acquire()
    c.t += 100  # long idle: refill is capped at capacity, not 100 tokens
    assert b.acquire() == 0 and b.acquire() == 0
    assert b.acquire() > 0


def test_backoff_grows_is_capped_and_jittered() -> None:
    assert backoff_delay(0, base=2, cap=60, rng=lambda: 1.0) == 2
    assert backoff_delay(3, base=2, cap=60, rng=lambda: 1.0) == 16
    assert backoff_delay(10, base=2, cap=60, rng=lambda: 1.0) == 60
    assert backoff_delay(3, rng=lambda: 0.5) == 8
    assert backoff_delay(3, rng=lambda: 0.0) == 0


class Boom(Exception):
    pass


def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom
        return "ok"

    out = retry_call(fn, is_retryable=lambda e: isinstance(e, Boom), max_attempts=5,
                     sleep=slept.append, rng=lambda: 1.0)
    assert out == "ok" and calls["n"] == 3 and slept == [2.0, 4.0]


def test_retry_gives_up_and_skips_non_retryable() -> None:
    n = {"n": 0}

    def always() -> None:
        n["n"] += 1
        raise Boom

    with pytest.raises(Boom):
        retry_call(always, is_retryable=lambda e: True, max_attempts=3, sleep=lambda s: None)
    assert n["n"] == 3
    n["n"] = 0
    with pytest.raises(Boom):
        retry_call(always, is_retryable=lambda e: False, max_attempts=3, sleep=lambda s: None)
    assert n["n"] == 1
