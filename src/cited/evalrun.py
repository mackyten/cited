"""Evaluation harness: retrieval ablation, end-to-end metrics, LLM-judge sample, threshold tuning."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cited.budget import enforce_budget
from cited.client import EMBED_BATCH, GeminiClient
from cited.config import Settings
from cited.generate import SYSTEM_PROMPT, format_prompt
from cited.pipeline import AskResult, Pipeline, RetrievedChunk
from cited.retrieval import STRATEGIES
from cited.store import Index

EVAL_DIR = Path("eval")
RETRIEVAL_DEPTH = 10
PREFERENCE = ("hybrid", "dense", "bm25")  # tie-break order when picking the best strategy

JUDGE_SYSTEM = """You are a strict fact-checker. You get numbered sources and an answer that cites them.
Decide whether EVERY factual claim in the answer is supported by the cited sources.
Reply with JSON only: {"verdict": "faithful" or "unfaithful", "unsupported": ["claim", ...]}"""


def norm(s: str) -> str:
    """Casefold and drop markdown emphasis/backticks so phrases match across formatting."""
    return re.sub(r"\s+", " ", re.sub(r"[`*]", "", s)).casefold()


def load_set(name: str) -> list[dict[str, Any]]:
    path = EVAL_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_rank(retrieved: list[RetrievedChunk], sources: list[str]) -> int | None:
    for r in retrieved:
        if r.chunk.source in sources:
            return r.n
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[k]


@dataclass
class RetrievalRow:
    id: str
    rank: int | None
    phrase_in_top5: bool


def retrieval_metrics(rows: list[RetrievalRow]) -> dict[str, float]:
    n = len(rows)
    return {
        "n": n,
        "hit@1": sum(r.rank == 1 for r in rows) / n,
        "hit@5": sum(r.rank is not None and r.rank <= 5 for r in rows) / n,
        "mrr": sum(1 / r.rank for r in rows if r.rank) / n,
        "phrase@5": sum(r.phrase_in_top5 for r in rows) / n,
    }


def eval_retrieval(pipe: Pipeline, questions: list[dict[str, Any]], strategy: str
                   ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[RetrievalRow] = []
    misses: list[dict[str, Any]] = []
    for q in questions:
        got = pipe.retrieve(q["question"], strategy, RETRIEVAL_DEPTH)
        rank = first_rank(got, q["sources"])
        phrase = any(norm(q["phrase"]) in norm(r.chunk.text + " " + " ".join(r.chunk.trail))
                     for r in got[:5])
        rows.append(RetrievalRow(q["id"], rank, phrase))
        if rank is None or rank > 5:
            misses.append({"id": q["id"], "question": q["question"], "expected": q["sources"],
                           "rank": rank,
                           "top3": [(r.chunk.source, round(r.score, 4)) for r in got[:3]]})
    return retrieval_metrics(rows), misses


def judge_answer(client: GeminiClient, res: AskResult) -> dict[str, Any]:
    prompt = (format_prompt(res.question, [r.chunk for r in res.retrieved]).rsplit("\n\nAnswer:", 1)[0]
              + f"\n\nAnswer to check:\n{res.answer}\n\nJSON verdict:")
    g = client.generate(JUDGE_SYSTEM, prompt)
    m = re.search(r"\{.*\}", g.text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {"verdict": d.get("verdict", "unparseable"), "unsupported": d.get("unsupported", []),
            "raw": g.text if not d else None}


def eval_end_to_end(pipe: Pipeline, questions: list[dict[str, Any]], strategy: str
                    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[dict[str, Any], AskResult]]]:
    results: list[tuple[dict[str, Any], AskResult]] = []
    pipe.live_queries = True
    for q in questions:
        results.append((q, pipe.ask(q["question"], strategy, pipe.s.top_k)))
    pipe.live_queries = False

    ans = [(q, r) for q, r in results if q["answerable"]]
    una = [(q, r) for q, r in results if not q["answerable"]]
    answered = [(q, r) for q, r in results if not r.abstained]
    failures: list[dict[str, Any]] = []

    correct = 0
    right_source = 0
    for q, r in ans:
        ok = (not r.abstained) and norm(q["phrase"]) in norm(r.answer)
        correct += ok
        if any(c["source"] in q["sources"] for c in r.citations):
            right_source += 1
        if not ok:
            failures.append(_failure(q, r, "answerable: " + (
                "abstained" if r.abstained else f"required phrase {q['phrase']!r} missing")))
    cite_ok = 0
    for q, r in answered:
        good = bool(r.citations) and not r.invented_citations
        cite_ok += good
        if not good:
            failures.append(_failure(q, r, f"citations: valid={[c['n'] for c in r.citations]} "
                                           f"invented={r.invented_citations}"))
    abst_ok = 0
    for q, r in una:
        abst_ok += r.abstained
        if not r.abstained:
            failures.append(_failure(q, r, "unanswerable but answered"))

    lat_all = [r.latency_ms for _, r in results]
    lat_model = [r.latency_ms for _, r in results if r.usage]
    metrics = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "answer_correct": correct / len(ans),
        "citation_valid": cite_ok / len(answered) if answered else float("nan"),
        "n_answered": len(answered),
        "cites_expected_source": right_source / len(ans),
        "abstain_accuracy": abst_ok / len(una),
        "false_abstention": sum(r.abstained for _, r in ans) / len(ans),
        "gate_abstentions": sum(r.abstain_reason is not None for _, r in results),
        "invented_citation_answers": sum(bool(r.invented_citations) for _, r in results),
        "latency_ms_p50": percentile(lat_all, 50), "latency_ms_p95": percentile(lat_all, 95),
        "latency_model_calls_p50": percentile(lat_model, 50),
        "latency_model_calls_p95": percentile(lat_model, 95),
    }
    return metrics, failures, results


def _failure(q: dict[str, Any], r: AskResult, why: str) -> dict[str, Any]:
    return {"id": q["id"], "why": why, "question": q["question"],
            "expected_sources": q.get("sources"), "expected_phrase": q.get("phrase"),
            "answer": r.answer, "abstain_reason": r.abstain_reason,
            "confidence": None if r.confidence is None else round(r.confidence, 4),
            "retrieved": [(x.chunk.source, round(x.score, 4)) for x in r.retrieved]}


def tune_threshold(pipe: Pipeline, questions: list[dict[str, Any]], strategy: str = "hybrid"
                   ) -> tuple[float, list[tuple[str, bool, float]]]:
    """Pick the gate threshold on the dev set: maximise balanced accuracy, ties -> lower (answer more)."""
    pts: list[tuple[str, bool, float]] = []
    for q in questions:
        got = pipe.retrieve(q["question"], strategy, pipe.s.top_k)
        conf = pipe._confidence(got)  # noqa: SLF001 - same value the live gate uses
        pts.append((q["id"], q["answerable"], conf if conf is not None else 0.0))
    vals = sorted({c for _, _, c in pts})
    cands = [vals[0] - 1e-3] + [(a + b) / 2 for a, b in zip(vals, vals[1:], strict=False)]
    n_a = sum(a for _, a, _ in pts)
    n_u = len(pts) - n_a

    def bal(t: float) -> float:
        tpr = sum(a and c >= t for _, a, c in pts) / n_a  # answerable let through
        tnr = sum((not a) and c < t for _, a, c in pts) / n_u  # unanswerable blocked
        return (tpr + tnr) / 2

    best = max(cands, key=lambda t: (round(bal(t), 6), -t))
    return round(best, 3), pts


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def run_eval_command(args: argparse.Namespace, settings: Settings, client: GeminiClient,
                     budget: int) -> int:
    echo = print
    index = Index.load(settings.index_dir)
    pipe = Pipeline(index, client, settings)
    questions = load_set(args.which)
    qtexts = [q["question"] for q in questions]

    # Stage 1: query embeddings (one batched request per 100 uncached questions).
    est1 = math.ceil(client.count_uncached_embeddings(qtexts, "RETRIEVAL_QUERY") / EMBED_BATCH)
    enforce_budget("eval: query embeddings", est1, budget, args.yes, echo)
    client.embed(qtexts, "RETRIEVAL_QUERY")

    if args.tune:
        t, pts = tune_threshold(pipe, questions)
        echo("\nTop dense similarity per dev question (hybrid retrieval):")
        for qid, ans, c in sorted(pts, key=lambda x: x[2]):
            echo(f"  {qid} {'answerable  ' if ans else 'unanswerable'} {c:.4f}")
        echo(f"\nsuggested abstain_threshold = {t}  (set CITED_ABSTAIN_THRESHOLD or Settings default)")
        return 0

    answerable = [q for q in questions if q["answerable"]]
    out: dict[str, Any] = {
        "date": dt.date.today().isoformat(), "set": args.which,
        "n_questions": len(questions), "text_model": settings.text_model,
        "embed_model": settings.embed_model, "embed_dim": settings.embed_dim,
        "top_k": settings.top_k, "abstain_threshold": settings.abstain_threshold,
        "corpus": index.meta.get("corpus"), "n_chunks": len(index.chunks),
    }

    # Retrieval ablation (no further API calls: query vectors are cached).
    ablation: dict[str, dict[str, float]] = {}
    misses: dict[str, list[dict[str, Any]]] = {}
    for s in STRATEGIES:
        ablation[s], misses[s] = eval_retrieval(pipe, answerable, s)
    out["retrieval"] = ablation
    echo(f"\n### Retrieval ablation ({len(answerable)} answerable questions, file-level hits)\n")
    echo(_table(["strategy", "hit@1", "hit@5", "MRR", "phrase in top-5"],
                [[s, _pct(m["hit@1"]), _pct(m["hit@5"]), f"{m['mrr']:.3f}", _pct(m["phrase@5"])]
                 for s, m in ablation.items()]))
    best = args.strategy or max(PREFERENCE, key=lambda s: (round(ablation[s]["mrr"], 6), -PREFERENCE.index(s)))
    out["best_strategy"] = best
    echo(f"\nbest by MRR: {best}")
    echo(f"\nRetrieval misses for {best} (expected file not in top 5):")
    for m in misses[best] or [None]:
        echo("  none" if m is None else f"  {m['id']} rank={m['rank']} expected={m['expected']} "
                                        f"top3={m['top3']} | {m['question']}")

    if args.retrieval_only:
        _save(out, args, best)
        return 0

    # Stage 2: live query embeddings (for honest latency) + uncached generations.
    pending = 0
    for q in questions:
        got = pipe.retrieve(q["question"], best, settings.top_k)
        conf = pipe._confidence(got)  # noqa: SLF001
        if pipe._gate(got, conf):  # noqa: SLF001
            continue
        if not client.is_generation_cached(SYSTEM_PROMPT, format_prompt(q["question"], [r.chunk for r in got])):
            pending += 1
    est2 = len(questions) + pending + args.judge
    enforce_budget(f"eval end-to-end ({best}): {len(questions)} live query embeds + "
                   f"{pending} generations + {args.judge} judge calls", est2, budget, args.yes, echo)

    req0, retries0 = client.requests_made, client.retries
    metrics, failures, results = eval_end_to_end(pipe, questions, best)
    out["end_to_end"] = {"strategy": best, **metrics}
    out["failures"] = failures
    echo(f"\n### End-to-end ({best}, top_k={settings.top_k}, abstain_threshold={settings.abstain_threshold})\n")
    echo(_table(["metric", "value"], [
        ["answer correct (required phrase present)", f"{_pct(metrics['answer_correct'])} of {metrics['n_answerable']}"],
        ["cites the expected source file", _pct(metrics["cites_expected_source"])],
        ["citation validity (>=1 valid, none invented)", f"{_pct(metrics['citation_valid'])} of {metrics['n_answered']} answers"],
        ["abstention accuracy on unanswerable", f"{_pct(metrics['abstain_accuracy'])} of {metrics['n_unanswerable']}"],
        ["false abstention on answerable", _pct(metrics["false_abstention"])],
        ["abstained by retrieval gate (no model call)", str(metrics["gate_abstentions"])],
        ["latency p50 / p95 (all questions)", f"{metrics['latency_ms_p50']:.0f} / {metrics['latency_ms_p95']:.0f} ms"],
        ["latency p50 / p95 (questions that called the model)",
         f"{metrics['latency_model_calls_p50']:.0f} / {metrics['latency_model_calls_p95']:.0f} ms"],
    ]))

    if args.judge:
        cand = [(q, r) for q, r in results if q["answerable"] and not r.abstained]
        step = max(1, len(cand) // args.judge)
        sample = cand[::step][:args.judge]
        verdicts = [{"id": q["id"], **judge_answer(client, r)} for q, r in sample]
        out["judge"] = {"model": settings.text_model, "n": len(verdicts), "verdicts": verdicts,
                        "faithful": sum(v["verdict"] == "faithful" for v in verdicts),
                        "note": "LLM judge (same model family as the generator); small sample."}
        echo(f"\nLLM-judge faithfulness (sample of {len(verdicts)}, judged by {settings.text_model}): "
             f"{out['judge']['faithful']}/{len(verdicts)} faithful")
        for v in verdicts:
            if v["verdict"] != "faithful":
                echo(f"  {v['id']}: {v['verdict']} {v['unsupported']}")

    out["api"] = {"requests": client.requests_made - req0, "retries": client.retries - retries0,
                  "rpm_limit_used": settings.rpm}
    echo(f"\n### Failures ({len(failures)})")
    for f in failures:
        echo(f"- {f['id']} [{f['why']}] {f['question']}\n    expected={f['expected_sources']} "
             f"phrase={f['expected_phrase']!r} confidence={f['confidence']} gate={f['abstain_reason']}\n"
             f"    retrieved={f['retrieved'][:3]}\n    answer={f['answer'][:300]!r}")
    echo(f"\nAPI requests this run: {out['api']['requests']} (retries {out['api']['retries']})")
    _save(out, args, best)
    return 0


def _save(out: dict[str, Any], args: argparse.Namespace, best: str) -> None:
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    mode = "retrieval" if args.retrieval_only else "full"
    path = d / f"{out['set']}-{mode}-{out['date']}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nresults written to {path}")


