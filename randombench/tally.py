"""Count each model's replies, and pick the model the next run goes to.

The collect-draws skill runs this before each run. Since version 9, a run
goes to the model with the fewest replies (tasks/collect.py), among the
models a run can draw from: MODELS lists it, and Kaggle lists it under the
same llm.name, with Model Proxy allowed. A tie goes to the first -m slug in
alphabetical order.

It prints a Markdown table of every model with stored draws or in MODELS,
fewest replies first, then the next model's -m slug on the last line. It asks
Kaggle for its model list, as `python -m randombench.models` does, so it's
read-only and prompts no model.

Usage: python -m randombench.tally [--data DIR]
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from randombench.extract import load_draws
from randombench.models import Listed, allowed_models, fetch_listed, stored_models
from randombench.page import DISPLAY_NAMES
from randombench.sync import ROOT


@dataclass(frozen=True)
class Row:
    """One model in the table."""

    llm_name: str
    name: str  # its display name
    slug: str  # its -m slug, or "" if Kaggle doesn't list it
    replies: int  # its stored draws with a reply
    why_not: str  # why a run can't draw from it, or "" if one can


def reply_counts(data_dir: Path) -> Counter:
    """Count each stored model's draws with a reply, by llm.name. A model whose
    every draw failed counts 0."""
    counts = Counter()
    for draw in load_draws(data_dir):
        counts[draw["model"]] += draw["reply"] is not None
    return counts


def tally(listed: list[Listed], stored: dict[str, str], replies: Counter, allowed: tuple[str, ...]) -> list[Row]:
    """Return a row for every model with stored draws or in MODELS, fewest replies first.

    stored maps each data/ folder's -m slug to the llm.name its newest run
    holds, as stored_models() returns it.
    """
    # A model's listing, found by the name Kaggle gives it now or, for a model
    # Kaggle has renamed or no longer names, by its data/ folder's slug.
    by_slug = {m.slug: m for m in listed}
    by_name = {name: by_slug[slug] for slug, name in stored.items() if slug in by_slug}
    by_name |= {m.llm_name: m for m in listed if m.llm_name}

    rows = []
    for name in set(replies) | set(allowed):
        m = by_name.get(name)
        if name not in allowed:
            why_not = "not in MODELS"
        elif m is None:
            why_not = "Kaggle doesn't list it"
        elif not m.proxy:
            why_not = "Model Proxy doesn't allow it"
        elif m.llm_name != name:
            # A run's llm.name would be the new name, which MODELS doesn't hold (D12).
            why_not = f"Kaggle renamed it {m.llm_name}"
        else:
            why_not = ""
        display = DISPLAY_NAMES[name][0] if name in DISPLAY_NAMES else (m.name if m else name)
        rows.append(Row(name, display, m.slug if m else "", replies[name], why_not))
    return sorted(rows, key=lambda r: (r.replies, r.slug, r.llm_name))


def next_model(rows: list[Row]) -> Row | None:
    """The model the next run goes to: the first a run can draw from."""
    return next((r for r in rows if not r.why_not), None)


def format_tally(rows: list[Row]) -> str:
    chosen = next_model(rows)
    lines = ["| # | Model | -m slug | Replies | Next run |", "|---:|---|---|---:|---|"]
    for i, r in enumerate(rows, 1):
        status = "next" if r is chosen else (f"never: {r.why_not}" if r.why_not else "")
        lines.append(f"| {i} | {r.name} | {r.slug or '-'} | {r.replies:,} | {status} |")
    drawable = sum(not r.why_not for r in rows)
    lines += ["", f"{sum(r.replies for r in rows):,} replies from {len(rows)} models,"
                  f" {drawable} of which a run can draw from.",
              f"Next: {chosen.slug if chosen else 'none'}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count each model's replies, and name the model the next run goes to."
        " Read-only: it prompts no model."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="where the stored draws are")
    args = parser.parse_args(argv)
    # A display name may hold a character outside Windows' default encoding for a pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = tally(fetch_listed(), stored_models(args.data), reply_counts(args.data), allowed_models())
    print(format_tally(rows))


if __name__ == "__main__":
    main()
