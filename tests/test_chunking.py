from cited.chunking import chunk_markdown, doc_url, slugify

BASE = "https://docs.test"
MD = """---
title: x
---
# Title { #title }

Intro paragraph that is long enough to survive the minimum chunk size filter easily.

## Section A { #sec-a }

Body of section A with enough words to count as a real chunk of documentation text.

```python
# not a heading
x = 1
```

### Sub A1

Nested content under A1 which should carry the full heading trail in its metadata.

{* ../../docs_src/skipped.py *}
"""


def test_heading_trail_anchor_and_url() -> None:
    chunks = chunk_markdown("tutorial/body.md", MD, base_url=BASE)
    sub = next(c for c in chunks if c.trail[-1] == "Sub A1")
    assert sub.trail == ("Title", "Section A", "Sub A1")
    assert sub.anchor == "sub-a1"
    assert sub.url == f"{BASE}/tutorial/body/#sub-a1"
    assert next(c for c in chunks if c.trail[-1] == "Section A").anchor == "sec-a"


def test_code_fence_is_not_a_heading_and_includes_dropped() -> None:
    chunks = chunk_markdown("a.md", MD, base_url=BASE)
    assert all("not a heading" not in c.trail for c in chunks)
    assert any("# not a heading" in c.text for c in chunks)
    assert all("docs_src" not in c.text for c in chunks)
    assert all("title: x" not in c.text for c in chunks)


def test_size_cap_and_overlap() -> None:
    para = " ".join(f"word{i}" for i in range(40))
    text = "# T\n\n" + "\n\n".join(f"{para} p{i}" for i in range(12))
    chunks = chunk_markdown("a.md", text, base_url=BASE, max_chars=500, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    # consecutive chunks of one section share a short tail
    assert any(w in chunks[1].text for w in chunks[0].text[-60:].split()[1:])


def test_ids_are_stable_and_unique() -> None:
    a = chunk_markdown("a.md", MD, base_url=BASE)
    b = chunk_markdown("a.md", MD, base_url=BASE)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_tiny_sections_are_dropped() -> None:
    assert chunk_markdown("a.md", "# T\n\nhi\n", base_url=BASE) == []


def test_helpers() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert doc_url("index.md", BASE, None) == f"{BASE}/"
    assert doc_url("advanced/index.md", BASE, "x") == f"{BASE}/advanced/#x"
