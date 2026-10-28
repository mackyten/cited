"""Small persistent key-value cache on SQLite (embeddings by content hash, eval responses)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


class DiskCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._db.commit()

    def get(self, key: str) -> bytes | None:
        row = self._db.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def set(self, key: str, value: bytes) -> None:
        self._db.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, value))
        self._db.commit()

    def __contains__(self, key: str) -> bool:
        return self._db.execute("SELECT 1 FROM kv WHERE k = ?", (key,)).fetchone() is not None

    def close(self) -> None:
        self._db.close()
