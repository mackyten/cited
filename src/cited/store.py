"""On-disk index: chunks.jsonl (metadata), vectors.npy (float32, unit-norm), meta.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cited.chunking import Chunk


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # (n, dim)
    meta: dict[str, object]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.to_json()) for c in self.chunks) + "\n")
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @staticmethod
    def load(directory: Path) -> "Index":
        if not (directory / "meta.json").is_file():
            raise FileNotFoundError(f"No index at {directory}. Run `cited ingest` first.")
        chunks = [Chunk.from_json(json.loads(line))
                  for line in (directory / "chunks.jsonl").read_text().splitlines() if line]
        return Index(chunks, np.load(directory / "vectors.npy"),
                     json.loads((directory / "meta.json").read_text()))
