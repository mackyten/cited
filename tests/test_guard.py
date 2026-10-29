from fastapi.testclient import TestClient

from cited.api import create_app
from cited.guard import DemoGuard, client_ip, guard_from_env


class Clock:
    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_per_ip_minute_limit_then_recovers() -> None:
    clock = Clock()
    g = DemoGuard(2, 100, 100, clock)
    assert g.check("a").allowed and g.check("a").allowed
    v = g.check("a")
    assert not v.allowed and 1 <= v.retry_after <= 60
    assert g.check("b").allowed  # other addresses are unaffected
    clock.t += 61
    assert g.check("a").allowed


def test_per_ip_daily_limit() -> None:
    clock = Clock()
    g = DemoGuard(100, 3, 100, clock)
    for _ in range(3):
        assert g.check("a").allowed
        clock.t += 120
    v = g.check("a")
    assert not v.allowed and "come back tomorrow" in v.message


def test_global_cap_resets_at_pacific_midnight() -> None:
    clock = Clock()
    g = DemoGuard(100, 100, 2, clock)
    assert g.check("a").allowed and g.check("b").allowed
    v = g.check("c")
    assert not v.allowed and v.retry_after > 0
    clock.t += v.retry_after + 1
    assert g.check("c").allowed


def test_rejected_requests_are_not_counted() -> None:
    g = DemoGuard(1, 100, 100, Clock())
    g.check("a")
    for _ in range(5):
        assert not g.check("a").allowed
    assert g._global == 1


def test_client_ip_trusts_only_the_proxy_written_entry() -> None:
    h = {"x-forwarded-for": "6.6.6.6, 203.0.113.9"}  # first entry is client-supplied
    assert client_ip(h, "10.0.0.1", 1) == "203.0.113.9"
    assert client_ip({}, "10.0.0.1", 1) == "10.0.0.1"
    assert client_ip(h, "10.0.0.1", 0) == "10.0.0.1"


def test_guard_is_off_unless_configured() -> None:
    assert guard_from_env({}) == (None, 0)
    g, hops = guard_from_env({"CITED_DEMO_GLOBAL_PER_DAY": "50"})
    assert g is not None and g.global_day == 50 and hops == 1


def test_api_returns_429_with_retry_after_and_message(pipeline) -> None:  # type: ignore[no-untyped-def]
    c = TestClient(create_app(pipeline, DemoGuard(1, 10, 10)))
    body = {"question": "Pydantic request body model"}
    assert c.post("/ask", json=body).status_code == 200
    r = c.post("/ask", json=body)
    assert r.status_code == 429 and "Too many" in r.json()["detail"] and int(r.headers["retry-after"]) >= 1
    assert c.post("/ask/stream", json=body).status_code == 429
    assert c.get("/health").status_code == 200  # health is never throttled
