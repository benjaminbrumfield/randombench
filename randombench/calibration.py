"""How the sub-scores vary by chance (PLAN.md 3.2, 3.3).

calibrate() simulates many runs of N answers from a known distribution and
summarizes each check's score over them. For a perfect generator, every
score should average about 1 at any N (D7), and the 95% range shows how far
a score moves by chance alone. For a made-up biased model, each score
should average the model's true score at any N. The tests check both.

A run's counts are drawn from a multinomial distribution, which is exactly
how the counts of N independent draws are distributed, without drawing each
one.

Usage: python -m randombench.calibration [N ...] [--trials T] [--seed S]
"""

import argparse
from dataclasses import dataclass

import numpy as np

from randombench.checks import CHECKS, N_VALUES

PERFECT = np.full(N_VALUES, 1 / N_VALUES)


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="How a perfect generator's sub-scores vary at each N.")
    parser.add_argument("n", type=int, nargs="*", default=[100, 300, 500, 600, 1200])
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    results = {n: calibrate(n, args.trials, args.seed) for n in args.n}
    print(f"A perfect generator's sub-scores: mean and 95% range over {args.trials:,} runs (seed {args.seed})")
    print(f"{'':16}" + "".join(f"{f'N = {n}':>22}" for n in args.n))
    for name in CHECKS:
        cells = [results[n][name] for n in args.n]
        print(f"{name:16}" + "".join(f"{f'{s.mean:.3f} ({s.low:.3f}-{s.high:.3f})':>22}" for s in cells))


if __name__ == "__main__":
    main()
