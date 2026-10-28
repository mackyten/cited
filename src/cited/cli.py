"""cited CLI: ingest, ask, eval, serve."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from cited.budget import BudgetExceeded, enforce_budget
from cited.cache import DiskCache
from cited.client import GeminiClient, GenaiBackend
from cited.config import MissingKeyError, Settings, load_settings, require_api_key
from cited.retrieval import STRATEGIES


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def make_client(settings: Settings) -> GeminiClient:
    backend = GenaiBackend(require_api_key(), settings)
    return GeminiClient(backend, settings, DiskCache(settings.cache_path), log=log)


def _add_budget_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--yes", action="store_true", help="proceed even if over the request budget")
    p.add_argument("--budget", type=int, default=None, help="max estimated API requests")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cited", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="download the docs, chunk, embed, and write the index")
    i.add_argument("--refresh-corpus", action="store_true", help="re-download the docs")
    _add_budget_args(i)

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--strategy", choices=STRATEGIES, default="hybrid")
    a.add_argument("--top-k", type=int, default=None)
    a.add_argument("--json", action="store_true", help="print the full result as JSON")

    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--set", dest="which", choices=("golden", "dev"), default="golden")
    e.add_argument("--retrieval-only", action="store_true", help="skip generation (fewest requests)")
    e.add_argument("--strategy", choices=STRATEGIES, default=None,
                   help="end-to-end strategy (default: best MRR)")
    e.add_argument("--judge", type=int, default=0, metavar="N",
                   help="LLM-judge faithfulness on N sampled answerable questions")
    e.add_argument("--tune", action="store_true", help="suggest an abstention threshold (use with --set dev)")
    e.add_argument("--out", default="eval/results", help="directory for result files")
    _add_budget_args(e)

    s = sub.add_parser("serve", help="run the FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    try:
        return _dispatch(args, settings)
    except (MissingKeyError, BudgetExceeded, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    budget = getattr(args, "budget", None) or settings.request_budget
    if args.cmd == "ingest":
        from cited.ingest import build_chunks, estimate_ingest_requests, run_ingest

        client = make_client(settings)
        _, chunks = build_chunks(settings, args.refresh_corpus)
        enforce_budget("ingest", estimate_ingest_requests(client, [c.embed_text for c in chunks]),
                       budget, args.yes, log)
        run_ingest(settings, client, refresh=False, echo=log)
        return 0

    if args.cmd == "ask":
        from cited.pipeline import Pipeline
        from cited.store import Index

        client = make_client(settings)
        pipe = Pipeline(Index.load(settings.index_dir), client, settings)
        r = pipe.ask(args.question, args.strategy, args.top_k)
        if args.json:
            print(json.dumps(r.to_json(), indent=2))
        else:
            print(r.answer)
            if r.abstain_reason:
                print(f"\n(abstained without calling the model: {r.abstain_reason})")
            for c in r.citations:
                print(f"  [{c['n']}] {c['source']} > {c['heading']}  {c['url']}")
            if r.invented_citations:
                print(f"  WARNING invented citations: {r.invented_citations}")
            log(f"{r.latency_ms:.0f} ms, confidence={r.confidence}, requests={client.requests_made}")
        return 0

    if args.cmd == "eval":
        from cited.evalrun import run_eval_command

        return run_eval_command(args, settings, make_client(settings), budget)

    if args.cmd == "serve":
        import uvicorn

        from cited.api import create_app
        from cited.pipeline import Pipeline
        from cited.store import Index

        pipe = Pipeline(Index.load(settings.index_dir), make_client(settings), settings)
        uvicorn.run(create_app(pipe), host=args.host, port=args.port)
        return 0
    return 1
