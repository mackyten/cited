from cited.config import ABSTAIN_SENTENCE


def test_answers_with_valid_citations(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    r = pipeline.ask("declare a Pydantic model request body", "hybrid", 3)
    assert not r.abstained and r.citations and not r.invented_citations
    assert r.citations[0]["n"] == 1 and r.citations[0]["source"] == r.retrieved[0].chunk.source
    assert backend.generate_calls == 1 and r.usage["prompt_tokens"] == 100


def test_invented_citation_is_reported(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    backend.answer = "Real [1] and fake [9]."
    r = pipeline.ask("Pydantic request body model", "bm25", 3)
    assert r.invented_citations == [9] and [c["n"] for c in r.citations] == [1]


def test_low_confidence_abstains_without_calling_the_model(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    pipeline.s = type(pipeline.s)(**{**pipeline.s.__dict__, "abstain_threshold": 0.99})
    r = pipeline.ask("completely unrelated xylophone question", "dense", 3)
    assert r.abstained and r.answer == ABSTAIN_SENTENCE and "below threshold" in r.abstain_reason
    assert backend.generate_calls == 0 and r.usage == {}


def test_bm25_has_no_confidence_gate(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    pipeline.s = type(pipeline.s)(**{**pipeline.s.__dict__, "abstain_threshold": 0.99})
    r = pipeline.ask("Pydantic model", "bm25", 3)
    assert r.confidence is None and backend.generate_calls == 1


def test_model_abstention_is_recognised(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    backend.answer = ABSTAIN_SENTENCE
    r = pipeline.ask("Pydantic request body model", "hybrid", 3)
    assert r.abstained and r.abstain_reason is None and r.citations == []


def test_stream_yields_retrieved_tokens_done(pipeline) -> None:  # type: ignore[no-untyped-def]
    evs = list(pipeline.ask_stream("Pydantic request body model", "hybrid", 3))
    assert [e["event"] for e in evs] == ["retrieved", "token", "token", "done"]
    assert evs[-1]["data"]["answer"] == "FastAPI uses Pydantic [1]."
    assert evs[-1]["data"]["citations"][0]["n"] == 1


def test_live_mode_bypasses_generation_cache(pipeline, backend) -> None:  # type: ignore[no-untyped-def]
    pipeline.ask("Pydantic request body model", "bm25", 3)
    pipeline.ask("Pydantic request body model", "bm25", 3)
    assert backend.generate_calls == 1  # second call served from the cache
    pipeline.live = True
    r = pipeline.ask("Pydantic request body model", "bm25", 3)
    assert backend.generate_calls == 2 and not r.cached
