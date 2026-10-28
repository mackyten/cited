"""Token-bucket rate limiter and exponential backoff with jitter (clock and sleep injectable)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class TokenBucket:
    def __init__(self, rate_per_min: float, capacity: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate = rate_per_min / 60.0  # tokens per second
        self.capacity = float(capacity)
        self._clock, self._sleep = clock, sleep
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self) -> float:
        """Block until a token is available. Returns seconds waited."""
        self._refill()
        waited = 0.0
        if self._tokens < 1.0:
            waited = (1.0 - self._tokens) / self.rate
            self._sleep(waited)
            self._refill()
        self._tokens -= 1.0
        return waited


def backoff_delay(attempt: int, base: float = 2.0, cap: float = 60.0,
                  rng: Callable[[], float] = random.random) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng() * min(cap, base * (2 ** attempt))


def retry_call(fn: Callable[[], T], *, is_retryable: Callable[[Exception], bool],
               max_attempts: int = 6, base: float = 2.0, cap: float = 60.0,
               sleep: Callable[[float], None] = time.sleep,
               rng: Callable[[], float] = random.random,
               on_retry: Callable[[int, Exception, float], None] | None = None) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified by is_retryable
            if attempt == max_attempts - 1 or not is_retryable(e):
                raise
            delay = backoff_delay(attempt, base, cap, rng)
            if on_retry:
                on_retry(attempt, e, delay)
            sleep(delay)
    raise AssertionError("unreachable")
