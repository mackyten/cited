"""Heading-aware markdown chunking with a size cap and small overlap."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
CUSTOM_ID_RE = re.compile(r"\s*\{\s*#([\w\-]+)\s*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INCLUDE_RE = re.compile(r"^\s*\{\*.*\*\}\s*$")  # code-include directives (source not fetched)
IMAGE_RE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str  # content hash of `embed_text`
    source: str  # path relative to the corpus root, e.g. "tutorial/body.md"
    trail: tuple[str, ...]  # heading trail
    anchor: str | None
    text: str  # body shown to the model and the user
    url: str | None

    @property
    def embed_text(self) -> str:
        """Heading context plus body: what gets embedded and BM25-indexed."""
        head = " > ".join(self.trail)
        return f"{head}\n\n{self.text}" if head else self.text

    def to_json(self) -> dict[str, object]:
        return {"id": self.id, "source": self.source, "trail": list(self.trail),
                "anchor": self.anchor, "text": self.text, "url": self.url}

    @staticmethod
    def from_json(d: dict[str, object]) -> "Chunk":
        return Chunk(id=str(d["id"]), source=str(d["source"]), trail=tuple(d["trail"]),  # type: ignore[arg-type]
                     anchor=d["anchor"], text=str(d["text"]), url=d["url"])  # type: ignore[arg-type]


def slugify(heading: str) -> str:
    s = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")


def doc_url(source: str, base: str, anchor: str | None) -> str:
    path = source[:-3] if source.endswith(".md") else source
    if path.endswith("index"):
        path = path[: -len("index")].rstrip("/")
    url = f"{base}/{path}/" if path else f"{base}/"
    return f"{url}#{anchor}" if anchor else url


def _clean(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_comment = False
    for line in lines:
        if "<!--" in line:
            in_comment = "-->" not in line
            continue
        if in_comment:
            in_comment = "-->" not in line
            continue
        if INCLUDE_RE.match(line) or IMAGE_RE.match(line):
            continue
        out.append(line.rstrip())
    return out


def _strip_frontmatter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return lines[i + 1:]
    return lines


def _sections(lines: list[str]) -> list[tuple[tuple[str, ...], str | None, list[str]]]:
    """Split into (heading trail, anchor, body lines); fenced code is never read as headings."""
    sections: list[tuple[tuple[str, ...], str | None, list[str]]] = []
    stack: list[tuple[int, str]] = []
    anchor: str | None = None
    body: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            sections.append((tuple(t for _, t in stack), anchor, body))
            level, raw = len(m.group(1)), m.group(2)
            cid = CUSTOM_ID_RE.search(raw)
            title = CUSTOM_ID_RE.sub("", raw).strip()
            anchor = cid.group(1) if cid else slugify(title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            body = []
        else:
            body.append(line)
    sections.append((tuple(t for _, t in stack), anchor, body))
    return sections


def _blocks(body: list[str]) -> list[str]:
    """Paragraph blocks; a fenced code block stays in one piece."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in body:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _pack(blocks: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedy-pack blocks up to max_chars. Consecutive pieces share a short tail."""
    pieces: list[str] = []
    cur = ""
    for b in blocks:
        while len(b) > max_chars:  # oversized block: hard-split on a line boundary
            cut = b.rfind("\n", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            head, b = b[:cut], b[cut:].lstrip("\n")
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(head)
        if cur and len(cur) + len(b) + 2 > max_chars:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if tail and " " in tail else tail
            cur = f"{tail}\n\n{b}" if tail and len(tail) + len(b) + 2 <= max_chars else b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        pieces.append(cur)
    return pieces


def chunk_markdown(source: str, text: str, *, base_url: str, max_chars: int = 1500,
                   overlap: int = 150, min_chars: int = 80) -> list[Chunk]:
    lines = _clean(_strip_frontmatter(text.splitlines()))
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for trail, anchor, body in _sections(lines):
        blocks = _blocks(body)
        if sum(len(b) for b in blocks) < min_chars:
            continue
        for piece in _pack(blocks, max_chars, overlap):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            probe = Chunk("", source, trail, anchor, piece, None)
            cid = hashlib.sha256(f"{source}\n{probe.embed_text}".encode()).hexdigest()[:16]
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(Chunk(cid, source, trail, anchor, piece,
                                doc_url(source, base_url, anchor)))
    return chunks


def chunk_corpus(root: Path, *, base_url: str, max_chars: int, overlap: int,
                 min_chars: int) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(root.rglob("*.md")):
        rel = md.relative_to(root).as_posix()
        out.extend(chunk_markdown(rel, md.read_text(encoding="utf-8"), base_url=base_url,
                                  max_chars=max_chars, overlap=overlap, min_chars=min_chars))
    return out
