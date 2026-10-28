"""retrieve -> (abstain gate) -> generate -> validate. One place, used by CLI, API and eval."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cited.chunking import Chunk
from cited.client import GeminiClient
from cited.config import ABSTAIN_SENTENCE, Settings
from cited.generate import (SYSTEM_PROMPT, CitationCheck, format_prompt, is_abstention,
                            validate_citations)
from cited.retrieval import STRATEGIES, Hit, Retriever, build_retriever
from cited.store import Index


@dataclass
class RetrievedChunk:
    n: int  # 1-based source number shown to the model
    chunk: Chunk
    score: float
    dense: float | None

    def to_json(self) -> dict[str, Any]:
        return {"n": self.n, "score": round(self.score, 5),
                "dense": None if self.dense is None else round(self.dense, 4),
                **{k: v for k, v in self.chunk.to_json().items() if k != "id"}, "id": self.chunk.id}


@dataclass
class AskResult:
    question: str
    strategy: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    citations: list[dict[str, Any]]
    invented_citations: list[int]
    retrieved: list[RetrievedChunk]
    latency_ms: float
    confidence: float | None
    cached: bool = False
    usage: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["retrieved"] = [r.to_json() for r in self.retrieved]
        return d


class Pipeline:
    def __init__(self, index: Index, client: GeminiClient, settings: Settings) -> None:
        self.index, self.client, self.s = index, client, settings
        self._retrievers: dict[str, Retriever] = {}

    def embed_query(self, q: str) -> np.ndarray:
        return self.client.embed([q], "RETRIEVAL_QUERY")[0]

    def retriever(self, strategy: str) -> Retriever:
        if strategy not in self._retrievers:
            self._retrievers[strategy] = build_retriever(
                strategy, self.index.chunks, self.index.vectors, self.embed_query)
        return self._retrievers[strategy]

    def retrieve(self, question: str, strategy: str, top_k: int) -> list[RetrievedChunk]:
        hits: list[Hit] = self.retriever(strategy).search(question, top_k)
        return [RetrievedChunk(n, self.index.chunks[h.index], h.score, h.dense)
                for n, h in enumerate(hits, start=1)]

    def _confidence(self, retrieved: list[RetrievedChunk]) -> float | None:
        """Best dense cosine among retrieved chunks. None for BM25-only (no gate applied)."""
        scores = [r.dense for r in retrieved if r.dense is not None]
        return max(scores) if scores else None

    def _gate(self, retrieved: list[RetrievedChunk], conf: float | None) -> str | None:
        if not retrieved:
            return "no chunks retrieved"
        if conf is not None and conf < self.s.abstain_threshold:
            return f"top dense similarity {conf:.3f} below threshold {self.s.abstain_threshold:.3f}"
        return None

    def _finish(self, question: str, strategy: str, retrieved: list[RetrievedChunk],
                conf: float | None, answer: str, t0: float, *, abstain_reason: str | None,
                cached: bool, usage: dict[str, int], latency_ms: float | None = None) -> AskResult:
        check: CitationCheck = validate_citations(answer, len(retrieved))
        abstained = abstain_reason is not None or is_abstention(answer)
        by_n = {r.n: r for r in retrieved}
        cites = [{"n": n, "source": by_n[n].chunk.source, "heading": " > ".join(by_n[n].chunk.trail),
                  "url": by_n[n].chunk.url, "chunk_id": by_n[n].chunk.id} for n in check.valid]
        return AskResult(question, strategy, answer, abstained, abstain_reason, cites,
                         check.invented, retrieved,
                         latency_ms if latency_ms is not None else (time.perf_counter() - t0) * 1000,
                         conf, cached, usage)

    def ask(self, question: str, strategy: str = "hybrid", top_k: int | None = None) -> AskResult:
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, strategy, top_k or self.s.top_k)
        conf = self._confidence(retrieved)
        if reason := self._gate(retrieved, conf):
            return self._finish(question, strategy, retrieved, conf, ABSTAIN_SENTENCE, t0,
                                abstain_reason=reason, cached=False, usage={})
        prompt = format_prompt(question, [r.chunk for r in retrieved])
        g = self.client.generate(SYSTEM_PROMPT, prompt)
        # Cached responses report the original call's latency plus this run's retrieval time.
        retrieval_ms = (time.perf_counter() - t0) * 1000
        lat = (retrieval_ms + g.latency_ms) if g.cached else None
        return self._finish(question, strategy, retrieved, conf, g.text.strip(), t0,
                            abstain_reason=None, cached=g.cached, latency_ms=lat,
                            usage={"prompt_tokens": g.prompt_tokens, "output_tokens": g.output_tokens})

    def ask_stream(self, question: str, strategy: str = "hybrid",
                   top_k: int | None = None) -> Iterator[dict[str, Any]]:
        """Yields {'event': 'retrieved'|'token'|'done', 'data': ...} dicts."""
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, strategy, top_k or self.s.top_k)
        conf = self._confidence(retrieved)
        yield {"event": "retrieved", "data": [r.to_json() for r in retrieved]}
        if reason := self._gate(retrieved, conf):
            yield {"event": "token", "data": ABSTAIN_SENTENCE}
            yield {"event": "done", "data": self._finish(
                question, strategy, retrieved, conf, ABSTAIN_SENTENCE, t0, abstain_reason=reason,
                cached=False, usage={}).to_json()}
            return
        parts: list[str] = []
        for piece in self.client.stream(SYSTEM_PROMPT,
                                        format_prompt(question, [r.chunk for r in retrieved])):
            parts.append(piece)
            yield {"event": "token", "data": piece}
        yield {"event": "done", "data": self._finish(
            question, strategy, retrieved, conf, "".join(parts).strip(), t0,
            abstain_reason=None, cached=False, usage={}).to_json()}


def check_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    return strategy


Embedder = Callable[[str], np.ndarray]
