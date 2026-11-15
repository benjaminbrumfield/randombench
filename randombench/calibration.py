"""How the sub-scores vary by chance (PLAN.md 3.2, 3.3, 4.5).

calibrate() simulates many runs of N answers from a known distribution and
summarizes each check's score over them. For a perfect generator, every
score should average about 1 at any N (D7), and the 95% range shows how far
a score moves by chance alone. For a made-up biased model, each score
should average the model's true score at any N. The tests check both.

coverage() checks the 95% intervals (4.5): over many simulated runs, how
often each score's interval holds the model's true score. It should be
about 95%.

A run's counts are drawn from a multinomial distribution, which is exactly
how the counts of N independent draws are distributed, without drawing each
one.

Usage: python -m randombench.calibration [N ...] [--trials T] [--seed S]
       python -m randombench.calibration --coverage [N ...] [--trials T] [--seed S]
"""

import argparse
from dataclasses import dataclass

import numpy as np

from randombench.checks import (
    CHECKS,
    EDGE,
    MULTIPLE_OF_5,
    MULTIPLE_OF_10,
    N_VALUES,
    PRIME,
    REPEATED_DIGITS,
    score_answers,
    score_intervals,
)

PERFECT = np.full(N_VALUES, 1 / N_VALUES)


def made_up(weights, rest):
    """A distribution: the given probabilities, and `rest` spread evenly over the other values."""
    p = np.zeros(N_VALUES)
    p[list(weights)] = list(weights.values())
    others = [v for v in range(N_VALUES) if v not in weights]
    p[others] += rest / len(others)
    return p


# Made-up models, from a perfect generator to one value (3.3).
MODELS = {
    "perfect": PERFECT,
    "near-perfect": made_up({42: 0.03, 37: 0.03, 7: 0.02}, 0.92),
    "prime-leaning": 0.7 * PERFECT + 0.3 * PRIME / PRIME.sum(),
    "llm-like": made_up({42: 0.20, 37: 0.15, 47: 0.10, 73: 0.10, 27: 0.05, 57: 0.05, 17: 0.05, 67: 0.05}, 0.25),
    "two-tied": made_up({42: 0.3, 37: 0.3}, 0.4),
    "edge-avoiding": made_up({v: 0.0 for v in np.flatnonzero(EDGE)}, 1.0),
    "one-value": made_up({42: 1.0}, 0.0),
}


def true_scores(p: np.ndarray, invalid: float = 0.0) -> dict[str, float]:
    """Every score for a model that answers from p, with a share `invalid` of
    its replies invalid. These are what the estimates aim at."""
    last_digits = p.reshape(10, 10).sum(axis=0)

    def gap(mask):
        return abs(p[mask].sum() - mask.mean())

    checks = {
        "uniformity": min(1, 1 / (p**2).sum() / N_VALUES),
        "primes": 1 - gap(PRIME),
        "multiples_of_10": 1 - gap(MULTIPLE_OF_10),
        "multiples_of_5": 1 - gap(MULTIPLE_OF_5),
        "favorite": 1 - (p.max() - 1 / N_VALUES) / (1 - 1 / N_VALUES),
        "edges": 1 - gap(EDGE),
        "repeated_digits": 1 - gap(REPEATED_DIGITS),
        "last_digit": min(1, 1 / (last_digits**2).sum() / 10),
    }
    valid = 1 - invalid
    return {"headline": valid * checks["uniformity"], "invalid_answers": valid, **checks}


@dataclass(frozen=True)
class Summary:
    """How one score varies across simulated runs."""

    mean: float
    low: float  # 2.5th percentile
    median: float
    high: float  # 97.5th percentile


def sample_counts(p: np.ndarray, n: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Return the counts of `trials` runs of n answers drawn from p, shaped (trials, 100)."""
    return rng.multinomial(n, p, size=trials)


def calibrate(n: int, trials: int = 10_000, seed: int = 0, p: np.ndarray = PERFECT) -> dict[str, Summary]:
    """Summarize every check's score over `trials` simulated runs of n answers from p."""
    counts = sample_counts(p, n, trials, np.random.default_rng(seed))
    summaries = {}
    for name, check in CHECKS.items():
        scores = check(counts)
        low, median, high = np.percentile(scores, [2.5, 50, 97.5])
        summaries[name] = Summary(float(scores.mean()), float(low), float(median), float(high))
    return summaries


def sample_replies(p: np.ndarray, n: int, invalid: float, rng: np.random.Generator) -> list[int | None]:
    """One run of n parsed replies: each one None with chance `invalid`, otherwise drawn from p."""
    counts = rng.multinomial(n, np.append((1 - invalid) * p, invalid))
    values = np.repeat(np.arange(N_VALUES), counts[:N_VALUES]).tolist()
    return values + [None] * int(counts[N_VALUES])


def coverage(
    p: np.ndarray, n: int, trials: int = 500, invalid: float = 0.0, n_resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """How often each score's 95% interval holds the model's true score, over
    `trials` simulated runs of n replies (4.5). Fewer resamples than a scoring
    task uses keep it quick. They make each interval's width a little noisier,
    not wider or narrower on average."""
    rng = np.random.default_rng(seed)
    truth = true_scores(p, invalid)
    held = dict.fromkeys(truth, 0)
    for _ in range(trials):
        answers = sample_replies(p, n, invalid, rng)
        scores = score_answers(answers)
        halves = score_intervals(answers, n_resamples, seed=int(rng.integers(2**32)))
        for name in held:
            # Up to rounding: an exact score with a ±0 interval holds the truth.
            held[name] += abs(scores[name] - truth[name]) <= halves[name] + 1e-9
    return {name: count / trials for name, count in held.items()}


def print_calibration(ns: list[int], trials: int, seed: int) -> None:
    results = {n: calibrate(n, trials, seed) for n in ns}
    print(f"A perfect generator's sub-scores: mean and 95% range over {trials:,} runs (seed {seed})")
    print(f"{'':16}" + "".join(f"{f'N = {n}':>22}" for n in ns))
    for name in CHECKS:
        cells = [results[n][name] for n in ns]
        print(f"{name:16}" + "".join(f"{f'{s.mean:.3f} ({s.low:.3f}-{s.high:.3f})':>22}" for s in cells))


# The made-up models whose intervals the coverage table checks, with their
# share of invalid replies. One has invalid replies, so the headline and the
# invalid-answer score vary too.
COVERAGE_MODELS = [(name, p, 0.0) for name, p in MODELS.items()] + [
    ("llm-like, 10% invalid", MODELS["llm-like"], 0.1),
]


def print_coverage(ns: list[int], trials: int, seed: int) -> None:
    names = list(true_scores(PERFECT))
    for n in ns:
        print(f"How often the 95% interval holds the true score, over {trials:,} runs of N = {n} (seed {seed})")
        print(f"{'':22}" + "".join(f"{name[:9]:>10}" for name in names))
        for label, p, invalid in COVERAGE_MODELS:
            held = coverage(p, n, trials, invalid, seed=seed)
            print(f"{label:22}" + "".join(f"{held[name]:>10.0%}" for name in names))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="How the sub-scores and their intervals vary by chance.")
    parser.add_argument("n", type=int, nargs="*", help="numbers of answers (default: 100 300 500 600 1200, or 624 with --coverage)")
    parser.add_argument("--coverage", action="store_true", help="check the 95%% intervals on made-up models instead")
    parser.add_argument("--trials", type=int, help="simulated runs (default: 10,000, or 500 with --coverage)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.coverage:
        print_coverage(args.n or [624], args.trials or 500, args.seed)
    else:
        print_calibration(args.n or [100, 300, 500, 600, 1200], args.trials or 10_000, args.seed)


if __name__ == "__main__":
    main()
