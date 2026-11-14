import numpy as np
import pytest

from randombench.calibration import PERFECT, calibrate, main, sample_counts
from randombench.checks import (
    CHECKS,
    EDGE,
    MULTIPLE_OF_5,
    MULTIPLE_OF_10,
    PRIME,
    REPEATED_DIGITS,
    favorite_share,
    repeat_rate,
)


def made_up(weights, rest):
    """A distribution: the given probabilities, and `rest` spread evenly over the other values."""
    p = np.zeros(100)
    p[list(weights)] = list(weights.values())
    others = [v for v in range(100) if v not in weights]
    p[others] += rest / len(others)
    return p


MODELS = {
    "perfect": PERFECT,
    "near-perfect": made_up({42: 0.03, 37: 0.03, 7: 0.02}, 0.92),
    "prime-leaning": 0.7 * PERFECT + 0.3 * PRIME / PRIME.sum(),
    "llm-like": made_up({42: 0.20, 37: 0.15, 47: 0.10, 73: 0.10, 27: 0.05, 57: 0.05, 17: 0.05, 67: 0.05}, 0.25),
    "two-tied": made_up({42: 0.3, 37: 0.3}, 0.4),
    "edge-avoiding": made_up({v: 0.0 for v in np.flatnonzero(EDGE)}, 1.0),
    "one-value": made_up({42: 1.0}, 0.0),
}


def true_scores(p):
    """Each check's score for the true distribution p, which the estimates aim at."""
    last_digits = p.reshape(10, 10).sum(axis=0)

    def gap(mask):
        return abs(p[mask].sum() - mask.mean())

    return {
        "uniformity": min(1, 1 / (p**2).sum() / 100),
        "primes": 1 - gap(PRIME),
        "multiples_of_10": 1 - gap(MULTIPLE_OF_10),
        "multiples_of_5": 1 - gap(MULTIPLE_OF_5),
        "favorite": 1 - (p.max() - 0.01) / 0.99,
        "edges": 1 - gap(EDGE),
        "repeated_digits": 1 - gap(REPEATED_DIGITS),
        "last_digit": min(1, 1 / (last_digits**2).sum() / 10),
    }


def test_made_up_models_are_distributions():
    for p in MODELS.values():
        assert p.sum() == pytest.approx(1) and (p >= 0).all()


# Simulation


def test_a_simulated_run_has_n_answers_over_100_values():
    counts = sample_counts(PERFECT, 600, 50, np.random.default_rng(0))
    assert counts.shape == (50, 100)
    assert (counts.sum(axis=1) == 600).all()


def test_the_same_seed_gives_the_same_summaries():
    assert calibrate(300, trials=1000, seed=7) == calibrate(300, trials=1000, seed=7)
    assert calibrate(300, trials=1000, seed=7) != calibrate(300, trials=1000, seed=8)


# The estimates are unbiased


@pytest.mark.parametrize("name", ["perfect", "llm-like", "two-tied"])
@pytest.mark.parametrize("n", [20, 600])
def test_repeat_rate_averages_the_true_rate(name, n):
    p = MODELS[name]
    counts = sample_counts(p, n, 20_000, np.random.default_rng(1))
    assert repeat_rate(counts).mean() == pytest.approx((p**2).sum(), rel=0.02)


@pytest.mark.parametrize("n", [20, 100, 600])
def test_favorite_share_averages_1_percent_for_a_perfect_generator(n):
    # The plain top share would average 2.2% at N = 600, and far more at N = 20.
    counts = sample_counts(PERFECT, n, 20_000, np.random.default_rng(2))
    assert favorite_share(counts).mean() == pytest.approx(0.01, rel=0.05)


# D7: a model's average score is its true score at any N


@pytest.mark.parametrize("n", [300, 600, 1200])
@pytest.mark.parametrize("name", list(MODELS))
def test_average_scores_match_the_true_scores(name, n):
    summaries = calibrate(n, trials=2000, seed=3, p=MODELS[name])
    expected = true_scores(MODELS[name])
    for check in CHECKS:
        assert summaries[check].mean == pytest.approx(expected[check], abs=0.02), check


def test_a_perfect_generator_scores_about_1_even_at_100():
    for check, s in calibrate(100).items():
        assert s.mean > 0.94, check


def test_each_summary_is_ordered():
    for s in calibrate(600, trials=2000).values():
        assert s.low <= s.median <= s.high


def test_main_prints_every_check(capsys):
    main(["600", "--trials", "200"])
    out = capsys.readouterr().out
    for name in CHECKS:
        assert name in out
