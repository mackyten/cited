from cited.cache import DiskCache, digest


def test_roundtrip_and_persistence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "c.sqlite"
    c = DiskCache(p)
    assert c.get("k") is None and "k" not in c
    c.set("k", b"v1")
    c.set("k", b"v2")
    c.close()
    c2 = DiskCache(p)
    assert c2.get("k") == b"v2" and "k" in c2


def test_digest_separates_parts() -> None:
    assert digest("ab", "c") != digest("a", "bc")
    assert digest("a") == digest("a")
