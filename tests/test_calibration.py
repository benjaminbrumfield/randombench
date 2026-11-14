from math import exp, lgamma, log

import numpy as np
import pytest

from randombench.calibration import calibrate, main, perfect_counts
from randombench.checks import EDGE, MULTIPLE_OF_5, MULTIPLE_OF_10, PRIME, REPEATED_DIGITS


def expected_gap(n, p):
    """E|X/n - p| for X ~ Binomial(n, p), summed exactly over the pmf."""

    def pmf(k):
        return exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p))

    return sum(pmf(k) * abs(k / n - p) for k in range(n + 1))


@pytest.fixture(scope="module")
def at_600():
    return calibrate(600)


def test_a_simulated_run_has_n_draws_over_100_values():
    counts = perfect_counts(600, 50, np.random.default_rng(0))
    assert counts.shape == (50, 100)
    assert (counts.sum(axis=1) == 600).all()


def test_the_same_seed_gives_the_same_summaries():
    assert calibrate(300, trials=1000, seed=7) == calibrate(300, trials=1000, seed=7)
    assert calibrate(300, trials=1000, seed=7) != calibrate(300, trials=1000, seed=8)


@pytest.mark.parametrize(
    "name, mask",
    [("primes", PRIME), ("multiples_of_10", MULTIPLE_OF_10), ("multiples_of_5", MULTIPLE_OF_5),
     ("edges", EDGE), ("repeated_digits", REPEATED_DIGITS)],
)
def test_a_share_averages_its_ideal(at_600, name, mask):
    # A perfect generator's expected share is the set's size over 100. Over
    # 10,000 runs the mean is within about 0.0002 of it.
    assert at_600[name].mean == pytest.approx(mask.mean(), abs=0.001)


def test_the_distances_average_their_exact_expectation(at_600):
    # Total variation distance is half the sum of each value's gap, so its
    # expectation is the sum of each value's expected gap, though the counts
    # depend on one another. One value's count is Binomial(n, 1/100), and
    # one last digit's is Binomial(n, 1/10).
    assert at_600["uniformity"].mean == pytest.approx(0.5 * 100 * expected_gap(600, 0.01), abs=0.001)
    assert at_600["last_digit"].mean == pytest.approx(0.5 * 10 * expected_gap(600, 0.1), abs=0.001)


def test_a_perfect_generator_is_measurably_off_at_600(at_600):
    assert at_600["uniformity"].low > 0.1
    assert at_600["favorite"].low > 0.015  # more than the ideal 0.01 in nearly every run


def test_the_noise_shrinks_as_n_grows():
    runs = [calibrate(n, trials=2000) for n in (100, 300, 600, 1200)]
    for name in ("uniformity", "favorite", "last_digit"):
        means = [r[name].mean for r in runs]
        assert means == sorted(means, reverse=True)
    for name in ("primes", "edges"):
        widths = [r[name].high - r[name].low for r in runs]
        assert widths == sorted(widths, reverse=True)


def test_each_summary_is_ordered(at_600):
    for s in at_600.values():
        assert s.low <= s.median <= s.high


def test_main_prints_every_measurement(capsys):
    main(["600", "--trials", "200"])
    out = capsys.readouterr().out
    for name in ("uniformity", "favorite", "last_digit"):
        assert name in out
