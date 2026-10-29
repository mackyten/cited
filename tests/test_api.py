import json

from fastapi.testclient import TestClient

from cited.api import create_app


def test_health(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = TestClient(create_app(pipeline)).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["chunks"] == 3


def test_ask_returns_answer_citations_chunks_latency(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = TestClient(create_app(pipeline)).post("/ask", json={"question": "Pydantic request body model", "top_k": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] and body["citations"] and len(body["retrieved"]) == 2
    assert {"score", "text", "source", "n"} <= set(body["retrieved"][0]) and body["latency_ms"] >= 0


def test_ask_validation(pipeline) -> None:  # type: ignore[no-untyped-def]
    c = TestClient(create_app(pipeline))
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "x", "strategy": "nope"}).status_code == 422
    assert c.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422


def test_stream_is_sse(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = TestClient(create_app(pipeline)).post("/ask/stream", json={"question": "Pydantic request body model"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [b.split("\n")[0].removeprefix("event: ") for b in r.text.strip().split("\n\n")]
    assert events[0] == "retrieved" and events[-1] == "done"
    done = r.text.strip().split("\n\n")[-1].split("data: ", 1)[1]
    assert json.loads(done)["answer"].startswith("FastAPI")


def test_title_and_page(pipeline) -> None:  # type: ignore[no-untyped-def]
    app = create_app(pipeline)
    assert app.title == "cited"
    assert "cited" in TestClient(app).get("/").text


def test_static_fonts_are_served(pipeline) -> None:  # type: ignore[no-untyped-def]
    r = TestClient(create_app(pipeline)).get("/static/fonts/barlow-400.woff2")
    assert r.status_code == 200 and r.content[:4] == b"wOF2"
