import pytest

from cited.budget import BudgetExceeded, enforce_budget
from cited.evalrun import RetrievalRow, norm, percentile, retrieval_metrics, tune_threshold


def test_percentile_nearest_rank() -> None:
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95 and percentile([7.0], 95) == 7.0


def test_retrieval_metrics() -> None:
    rows = [RetrievalRow("a", 1, True), RetrievalRow("b", 3, True), RetrievalRow("c", None, False),
            RetrievalRow("d", 7, False)]
    m = retrieval_metrics(rows)
    assert m["hit@1"] == 0.25 and m["hit@5"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 7) / 4)


def test_norm_ignores_markdown_and_case() -> None:
    assert norm("The `Query(max_length=50)` **IS**  here") == "the query(max_length=50) is here"


def test_tune_threshold_splits_the_gap(pipeline) -> None:  # type: ignore[no-untyped-def]
    qs = [{"id": "a", "question": "Pydantic request body model", "answerable": True},
          {"id": "u", "question": "zzz qqq www", "answerable": False}]
    t, pts = tune_threshold(pipeline, qs, "dense")
    conf = {i: c for i, _, c in pts}
    assert conf["u"] < t <= conf["a"]


def test_budget_guard() -> None:
    enforce_budget("x", 10, 10, False, lambda m: None)
    with pytest.raises(BudgetExceeded):
        enforce_budget("x", 11, 10, False, lambda m: None)
    enforce_budget("x", 11, 10, True, lambda m: None)  # --yes
