"""Three interchangeable retrievers behind one interface: BM25, dense, hybrid (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cited.chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or that the this to "
    "what when where which with you your".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. snake_case identifiers also emit their parts (response_model)."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if "_" in tok:
            parts = [p for p in tok.split("_") if p]
            out.append(tok)
            out.extend(p for p in parts if p not in STOPWORDS)
        elif tok not in STOPWORDS:
            out.append(tok)
    return out


@dataclass(frozen=True)
class Hit:
    index: int  # position in the Index.chunks list
    score: float  # strategy-specific (bm25 score, cosine, or RRF score)
    dense: float | None = None  # cosine similarity, when a dense score exists


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over an in-memory inverted index."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=np.float64)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(tokenize(query)):
            for i in self.postings.get(t, ()):
                f = self.tf[i][t]
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s


def _top(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    return sorted(idx.tolist(), key=lambda i: (-scores[i], i))


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.bm25 = BM25([c.embed_text for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        s = self.bm25.scores(query)
        return [Hit(i, float(s[i])) for i in _top(s, k) if s[i] > 0]


class DenseRetriever:
    name = "dense"

    def __init__(self, vectors: np.ndarray, embed_query: Callable[[str], np.ndarray]) -> None:
        self.vectors, self.embed_query = vectors, embed_query

    def search(self, query: str, k: int) -> list[Hit]:
        sims = self.vectors @ self.embed_query(query)
        return [Hit(i, float(sims[i]), float(sims[i])) for i in _top(sims, k)]


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    name = "hybrid"

    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, depth: int = 50,
                 rrf_k: int = 60) -> None:
        self.sparse, self.dense, self.depth, self.rrf_k = sparse, dense, depth, rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        sp = self.sparse.search(query, self.depth)
        de = self.dense.search(query, self.depth)
        dense_score = {h.index: h.dense for h in de}
        fused = rrf([[h.index for h in sp], [h.index for h in de]], self.rrf_k)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        # Dense cosine for every returned chunk (needed for the abstention gate).
        sims = self.dense.vectors @ self.dense.embed_query(query)
        return [Hit(i, fused[i], float(dense_score.get(i, sims[i]))) for i in order]


STRATEGIES = ("bm25", "dense", "hybrid")


def build_retriever(strategy: str, chunks: Sequence[Chunk], vectors: np.ndarray,
                    embed_query: Callable[[str], np.ndarray]) -> Retriever:
    if strategy == "bm25":
        return BM25Retriever(chunks)
    dense = DenseRetriever(vectors, embed_query)
    if strategy == "dense":
        return dense
    if strategy == "hybrid":
        return HybridRetriever(BM25Retriever(chunks), dense)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
