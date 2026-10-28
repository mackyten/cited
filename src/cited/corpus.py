"""Fetch the FastAPI English docs (MIT) via a shallow, docs-only clone."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from cited.config import Settings

# Pages that are not prose documentation (changelogs, people lists, test pages).
EXCLUDE_NAMES = {
    "release-notes.md",
    "_llm-test.md",
    "fastapi-people.md",
    "management.md",
    "management-tasks.md",
    "newsletter.md",
    "external-links.md",
    "translation-banner.md",
}
EXCLUDE_DIRS = {"reference", "css", "js", "img"}  # reference/ is mkdocstrings stubs only


def fetch_corpus(settings: Settings, force: bool = False) -> Path:
    dest = settings.corpus_dir
    if dest.is_dir() and any(dest.rglob("*.md")) and not force:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "repo"
        run = lambda *a, cwd=None: subprocess.run(  # noqa: E731
            a, cwd=cwd, check=True, capture_output=True, text=True
        )
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            settings.corpus_repo, str(clone))
        run("git", "sparse-checkout", "set", settings.corpus_subdir, cwd=clone)
        sha = run("git", "rev-parse", "HEAD", cwd=clone).stdout.strip()
        src = clone / settings.corpus_subdir
        for md in src.rglob("*.md"):
            rel = md.relative_to(src)
            if rel.name in EXCLUDE_NAMES or EXCLUDE_DIRS & set(rel.parts):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(md, out)
    (dest / "_corpus.json").write_text(json.dumps({
        "repo": settings.corpus_repo, "subdir": settings.corpus_subdir,
        "commit": sha, "license": "MIT",
    }, indent=2))
    return dest
