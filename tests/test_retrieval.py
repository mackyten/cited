import numpy as np

from cited.retrieval import BM25, BM25Retriever, build_retriever, rrf, tokenize
from conftest import bow_vector


def test_tokenize_splits_snake_case_and_drops_stopwords() -> None:
    assert tokenize("The response_model is here") == ["response_model", "response", "model", "here"]


def test_bm25_ranks_relevant_doc_first() -> None:
    docs = ["cats purr softly", "dogs bark loudly at night", "the stock market fell"]
    s = BM25(docs).scores("dogs bark")
    assert int(np.argmax(s)) == 1
    assert s[2] == 0


def test_bm25_rare_terms_weigh_more() -> None:
    docs = ["common common common rare", "common common common common"]
    s = BM25(docs).scores("common rare")
    assert s[0] > s[1]


def test_rrf_math_and_order() -> None:
    fused = rrf([[1, 2, 3], [3, 2, 1]], k=60)
    assert fused[2] == 2 / 62
    assert fused[1] == 1 / 61 + 1 / 63
    assert fused[1] == fused[3]


def test_rrf_rewards_agreement() -> None:
    fused = rrf([[10, 11], [12, 10]], k=60)
    assert max(fused, key=fused.get) == 10


def test_hybrid_and_dense_find_the_right_chunk(index) -> None:  # type: ignore[no-untyped-def]
    embed = lambda q: np.asarray(bow_vector(q)) / np.linalg.norm(bow_vector(q))  # noqa: E731
    for strat in ("bm25", "dense", "hybrid"):
        r = build_retriever(strat, index.chunks, index.vectors, embed)
        hits = r.search("allowed origins CORSMiddleware", 2)
        assert index.chunks[hits[0].index].source == "tutorial/cors.md", strat


def test_bm25_returns_nothing_without_overlap(index) -> None:  # type: ignore[no-untyped-def]
    assert BM25Retriever(index.chunks).search("zzzqqq", 3) == []


def test_unknown_strategy() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_retriever("nope", [], np.zeros((0, 2)), lambda q: np.zeros(2))
