"""Pre-run request estimates. Refuse to start over budget unless --yes."""

from __future__ import annotations

from collections.abc import Callable


class BudgetExceeded(RuntimeError):
    pass


def enforce_budget(label: str, estimated: int, budget: int, yes: bool,
                   echo: Callable[[str], None] = print) -> None:
    echo(f"[budget] {label}: ~{estimated} API requests (budget {budget})")
    if estimated > budget and not yes:
        raise BudgetExceeded(
            f"{label} would make ~{estimated} requests, over the budget of {budget}. "
            "Re-run with --yes to proceed, or raise --budget / CITED_BUDGET.")
