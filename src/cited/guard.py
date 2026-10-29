"""Demo abuse guard: per-IP and global request caps for a public deployment.

State is in memory, so run a single instance (Cloud Run: --max-instances 1). Limits are
off unless the CITED_DEMO_* variables are set, so local use is never throttled.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

MINUTE, DAY = 60.0, 86400.0
# The Gemini free-tier daily quota resets at midnight Pacific, so the global cap does too.
QUOTA_TZ = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    message: str = ""
    retry_after: int = 0


class DemoGuard:
    def __init__(self, per_ip_minute: int, per_ip_day: int, global_day: int,
                 clock: Callable[[], float] = time.time) -> None:
        self.per_ip_minute, self.per_ip_day, self.global_day = per_ip_minute, per_ip_day, global_day
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._day_key = ""
        self._global = 0
        self._lock = threading.Lock()

    def _quota_day(self, now: float) -> str:
        return datetime.fromtimestamp(now, QUOTA_TZ).strftime("%Y-%m-%d")

    def _seconds_to_quota_reset(self, now: float) -> int:
        t = datetime.fromtimestamp(now, QUOTA_TZ)
        return int(86400 - (t.hour * 3600 + t.minute * 60 + t.second))

    def check(self, ip: str) -> Verdict:
        """Count one question from `ip` if it is allowed. Rejected requests are not counted."""
        now = self._clock()
        with self._lock:
            day = self._quota_day(now)
            if day != self._day_key:
                self._day_key, self._global = day, 0
            if self._global >= self.global_day:
                return Verdict(False, "The demo has reached its daily limit and resets at midnight "
                                      "Pacific time. You can run cited yourself; see the README.",
                               self._seconds_to_quota_reset(now))
            q = self._hits.setdefault(ip, deque())
            while q and q[0] <= now - DAY:
                q.popleft()
            if len(q) >= self.per_ip_day:
                return Verdict(False, f"You have used your {self.per_ip_day} demo questions for "
                                      "today. Please come back tomorrow.", math.ceil(q[0] + DAY - now))
            recent = [t for t in q if t > now - MINUTE]
            if len(recent) >= self.per_ip_minute:
                return Verdict(False, "Too many questions in a minute. Please wait a moment.",
                               math.ceil(recent[0] + MINUTE - now))
            q.append(now)
            self._global += 1
            if len(self._hits) > 5000:  # bound memory under a flood of distinct addresses
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - DAY}
            return Verdict(True)


def client_ip(headers: Mapping[str, str], peer: str, trusted_hops: int) -> str:
    """The caller's address behind `trusted_hops` proxies we control (Cloud Run: 1).

    Proxies append to X-Forwarded-For, so only the entry written by our own proxy is
    trustworthy; anything to its left is client-supplied and can be forged.
    """
    parts = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if trusted_hops > 0 and len(parts) >= trusted_hops:
        return parts[-trusted_hops]
    return peer


def guard_from_env(env: Mapping[str, str] | None = None) -> tuple[DemoGuard | None, int]:
    """(guard, trusted_hops). No guard unless CITED_DEMO_GLOBAL_PER_DAY is set."""
    env = os.environ if env is None else env
    if not env.get("CITED_DEMO_GLOBAL_PER_DAY"):
        return None, 0
    guard = DemoGuard(int(env.get("CITED_DEMO_IP_PER_MIN", "4")),
                      int(env.get("CITED_DEMO_IP_PER_DAY", "20")),
                      int(env["CITED_DEMO_GLOBAL_PER_DAY"]))
    return guard, int(env.get("CITED_TRUSTED_HOPS", "1"))
