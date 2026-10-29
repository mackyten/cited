"""FastAPI app: POST /ask, POST /ask/stream (SSE), GET /health, and a static page at /."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cited.pipeline import Pipeline
from cited.retrieval import STRATEGIES

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="cited", version="0.1.0",
                  description="Grounded Q&A over technical docs with citations and abstention.")

    def checked(req: AskRequest) -> None:
        if req.strategy not in STRATEGIES:
            raise HTTPException(422, f"strategy must be one of {list(STRATEGIES)}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "chunks": len(pipeline.index.chunks),
                "text_model": pipeline.s.text_model, "embed_model": pipeline.s.embed_model}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        checked(req)
        return pipeline.ask(req.question, req.strategy, req.top_k).to_json()

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        checked(req)

        def events() -> Iterator[str]:
            for ev in pipeline.ask_stream(req.question, req.strategy, req.top_k):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")  # fonts for the page

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
