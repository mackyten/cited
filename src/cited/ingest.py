"""Fetch -> chunk -> embed (cached, resumable) -> persist."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from cited.chunking import chunk_corpus
from cited.client import EMBED_BATCH, GeminiClient
from cited.config import Settings
from cited.corpus import fetch_corpus
from cited.store import Index


def build_chunks(settings: Settings, refresh: bool = False):  # type: ignore[no-untyped-def]
    root = fetch_corpus(settings, force=refresh)
    return root, chunk_corpus(root, base_url=settings.docs_base_url,
                              max_chars=settings.max_chunk_chars,
                              overlap=settings.overlap_chars, min_chars=settings.min_chunk_chars)


def estimate_ingest_requests(client: GeminiClient, texts: list[str]) -> int:
    return math.ceil(client.count_uncached_embeddings(texts, "RETRIEVAL_DOCUMENT") / EMBED_BATCH)


def run_ingest(settings: Settings, client: GeminiClient, *, refresh: bool = False,
               echo: Callable[[str], None] = print) -> Index:
    root, chunks = build_chunks(settings, refresh)
    echo(f"{len(chunks)} chunks from {len({c.source for c in chunks})} files")
    vectors = client.embed([c.embed_text for c in chunks], "RETRIEVAL_DOCUMENT")
    corpus_meta = json.loads((root / "_corpus.json").read_text()) if (root / "_corpus.json").is_file() else {}
    index = Index(chunks, vectors, {
        "embed_model": settings.embed_model, "embed_dim": settings.embed_dim,
        "n_chunks": len(chunks), "corpus": corpus_meta,
        "chunking": {"max_chars": settings.max_chunk_chars, "overlap": settings.overlap_chars},
    })
    index.save(settings.index_dir)
    echo(f"index written to {settings.index_dir} ({client.requests_made} API requests)")
    return index
