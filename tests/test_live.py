"""Live tests: need a real key. Skipped otherwise. Each makes one tiny request."""

import pytest

from cited.config import Settings, load_api_key

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(load_api_key() is None, reason="no Gemini API key in environment")]


def test_embedding_and_generation_roundtrip() -> None:
    from cited.client import GeminiClient, GenaiBackend

    s = Settings()
    c = GeminiClient(GenaiBackend(load_api_key() or "", s), s)
    v = c.embed(["hello"], "RETRIEVAL_QUERY")
    assert v.shape == (1, s.embed_dim)
    assert c.generate("Reply with one word.", "Say ok", use_cache=False).text.strip()
