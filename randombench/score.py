"""Score every model's stored draws (PLAN.md 3.4).

Loads the records in data/ (2.5), merges each model's draws across its runs
(D3), parses each reply (2.2) and scores the answers (3.3), each score with
its 95% interval (4.5). A failed call has no reply, so it's left out and
doesn't count as an invalid answer, but the table shows how many there were.
This runs offline only. A scoring task does the same parse-then-score step
itself (4.5).

Usage: python -m randombench.score [--data data]
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from randombench.checks import score_answers, score_intervals
from randombench.extract import load_draws
from randombench.parsing import parse_reply


@dataclass(frozen=True)
class ModelScores:
    model: str
    draws: int
    failed: int  # calls that returned no reply
    invalid: int  # replies the parser rejected
    valid: int  # the N the distribution checks score
    scores: dict[str, float | None]
    intervals: dict[str, float | None]  # each score's ± 95% half-width


def score_draws(draws: list[dict]) -> list[ModelScores]:
    """Score each model's draws, merged across runs. Models are in name order."""
    by_model = defaultdict(list)
    for draw in draws:
        by_model[draw["model"]].append(draw)

    results = []
    for model in sorted(by_model):
        model_draws = by_model[model]
        answers = [parse_reply(d["reply"]) for d in model_draws if d["reply"] is not None]
        invalid = sum(a is None for a in answers)
        results.append(
            ModelScores(
                model=model,
                draws=len(model_draws),
                failed=len(model_draws) - len(answers),
                invalid=invalid,
                valid=len(answers) - invalid,
                scores=score_answers(answers) if answers else {},
                intervals=score_intervals(answers) if answers else {},
            )
        )
    return results


# Column headings for the table, in score_answers' order.
_HEADINGS = {
    "headline": "HEADLINE",
    "invalid_answers": "invalid",
    "uniformity": "uniform",
    "primes": "primes",
    "multiples_of_10": "×10",
    "multiples_of_5": "×5",
    "favorite": "favorite",
    "edges": "edges",
    "repeated_digits": "repeats",
    "last_digit": "last",
}


def format_table(results: list[ModelScores]) -> str:
    """The scores, then their 95% intervals in the same columns."""
    width = max([len("model"), *(len(r.model) for r in results)])
    counts = ["draws", "failed", "invalid", "N"]
    lines = [f"{'model':{width}}" + "".join(f"{c:>8}" for c in counts) + "  │"
             + "".join(f"{h:>9}" for h in _HEADINGS.values())]
    for r in results:
        cells = [r.draws, r.failed, r.invalid, r.valid]
        scores = [r.scores.get(name) for name in _HEADINGS]
        lines.append(
            f"{r.model:{width}}" + "".join(f"{c:>8}" for c in cells) + "  │"
            + "".join(f"{'—' if s is None else f'{s:.3f}':>9}" for s in scores)
        )
    left = width + 8 * len(counts)  # the columns before the scores
    lines += [f"{'':{left}}  │", f"{'± 95% interval':{left}}  │"]
    for r in results:
        halves = [r.intervals.get(name) for name in _HEADINGS]
        lines.append(f"{r.model:{left}}  │" + "".join(f"{'—' if h is None else f'±{h:.3f}':>9}" for h in halves))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score every model's stored draws.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="where the records are")
    args = parser.parse_args(argv)
    results = score_draws(load_draws(args.data))
    if not results:
        print(f"no draws in {args.data}")
        return
    print(format_table(results))
    print("\nScores are in [0, 1], about 1 for a perfect generator. The headline is the valid"
          " share times uniformity. N counts valid answers; the checks after 'invalid' score"
          " only those, and they and the headline show — below 2. Each interval is 1.96 times"
          " the score's spread over 10,000 bootstrap resamples of the model's replies.")


if __name__ == "__main__":
    main()
