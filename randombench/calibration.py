"""What a perfect generator measures (PLAN.md 3.2).

A perfect generator gives every value probability 1/100, yet any finite run
of it strays from the ideal. At N = 600 its favorite value takes about 2.2%
of the answers, not 1%, and its distribution is about 0.16 from uniform.
calibrate() simulates many runs of N perfect draws and summarizes each
measurement in checks.py, so a model can be compared with a perfect
generator at the model's own N: the number of its valid answers, which
invalid replies push below 600.

A run's counts are drawn from a multinomial distribution, which is exactly
how the counts of N independent uniform draws are distributed, without
drawing each one.

Usage: python -m randombench.calibration [N ...] [--trials T] [--seed S]
"""

import argparse
from dataclasses import dataclass

import numpy as np

from randombench.checks import MEASUREMENTS, N_VALUES


@dataclass(frozen=True)
class Summary:
    """How one measurement varies across runs of a perfect generator."""

    mean: float
    low: float  # 2.5th percentile
    median: float
    high: float  # 97.5th percentile


def perfect_counts(n: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Return the counts of `trials` runs of n perfect draws each, shaped (trials, 100)."""
    return rng.multinomial(n, np.full(N_VALUES, 1 / N_VALUES), size=trials)


def calibrate(n: int, trials: int = 10_000, seed: int = 0) -> dict[str, Summary]:
    """Summarize every measurement over `trials` simulated runs of n perfect draws."""
    counts = perfect_counts(n, trials, np.random.default_rng(seed))
    summaries = {}
    for name, measure in MEASUREMENTS.items():
        values = measure(counts)
        low, median, high = np.percentile(values, [2.5, 50, 97.5])
        summaries[name] = Summary(float(values.mean()), float(low), float(median), float(high))
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="What a perfect generator measures at each N.")
    parser.add_argument("n", type=int, nargs="*", default=[100, 300, 500, 600, 1200])
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    results = {n: calibrate(n, args.trials, args.seed) for n in args.n}
    print(f"Mean and 95% range over {args.trials:,} runs of a perfect generator (seed {args.seed})")
    print(f"{'':16}" + "".join(f"{f'N = {n}':>22}" for n in args.n))
    for name in MEASUREMENTS:
        cells = [results[n][name] for n in args.n]
        print(f"{name:16}" + "".join(f"{f'{s.mean:.3f} ({s.low:.3f}-{s.high:.3f})':>22}" for s in cells))


if __name__ == "__main__":
    main()
