import numpy as np
import pytest

from randombench.checks import (
    CHECKS,
    EDGE,
    MULTIPLE_OF_5,
    MULTIPLE_OF_10,
    PRIME,
    REPEATED_DIGITS,
    counts_of,
    effective_values,
    favorite_share,
    last_digit_counts,
    primes,
    repeat_rate,
    share_gap,
    uniformity,
)

# Value sets


def test_value_sets_have_the_sizes_in_the_bias_table():
    sizes = [PRIME.sum(), MULTIPLE_OF_10.sum(), MULTIPLE_OF_5.sum(), EDGE.sum(), REPEATED_DIGITS.sum()]
    assert sizes == [25, 10, 20, 20, 9]


def test_value_sets_hold_the_right_values():
    primes_ = list(np.flatnonzero(PRIME))
    assert primes_[:6] == [2, 3, 5, 7, 11, 13]
    assert primes_[-3:] == [83, 89, 97]
    assert list(np.flatnonzero(MULTIPLE_OF_10)) == list(range(0, 100, 10))
    assert list(np.flatnonzero(MULTIPLE_OF_5)) == list(range(0, 100, 5))
    assert list(np.flatnonzero(EDGE)) == list(range(10)) + list(range(90, 100))
    assert list(np.flatnonzero(REPEATED_DIGITS)) == list(range(11, 100, 11))


# Counting


def test_counts_of_counts_each_value():
    counts = counts_of([3, 3, 47])
    assert counts.shape == (100,)
    assert (counts[3], counts[47], counts.sum()) == (2, 1, 3)


@pytest.mark.parametrize("bad", [[100], [-1], [3, 250]])
def test_counts_of_rejects_values_outside_the_range(bad):
    with pytest.raises(ValueError):
        counts_of(bad)


def test_last_digit_counts():
    assert list(last_digit_counts(counts_of([7, 17, 47, 50, 3]))) == [1, 0, 0, 1, 0, 0, 0, 3, 0, 0]


# The estimates


def test_repeat_rate_is_the_share_of_equal_pairs():
    assert float(repeat_rate(counts_of([42] * 12))) == 1
    assert float(repeat_rate(counts_of([1, 2, 3]))) == 0
    # Six pairs, two of them equal (7 with 7, 8 with 8).
    assert float(repeat_rate(counts_of([7, 7, 8, 8]))) == pytest.approx(1 / 3)


def test_effective_values():
    assert float(effective_values(counts_of([7, 7, 8, 8]))) == pytest.approx(3)
    assert float(effective_values(counts_of([42] * 12))) == 1
    assert effective_values(counts_of([1, 2, 3])) == np.inf  # no repeats yet


def test_share_gap_has_the_noise_removed():
    # All prime: the observed share has no spread, so the gap is the plain gap.
    assert float(share_gap(counts_of([2, 3, 5, 7, 11] * 2), PRIME)) == pytest.approx(0.75)
    assert float(share_gap(counts_of([42] * 10), PRIME)) == pytest.approx(0.25)
    # Exactly 25% primes: the noise estimate makes the squared gap negative, floored at 0.
    assert float(share_gap(counts_of([2, 4, 6, 8] * 3), PRIME)) == 0
    # 50% primes in 4 answers: 0.25^2 - 0.5 * 0.5 / 3 < 0. Four answers can't show the bias.
    assert float(share_gap(counts_of([2, 3, 4, 6]), PRIME)) == 0
    # 50% primes in 400 answers: sqrt(0.0625 - 0.25/399).
    assert float(share_gap(counts_of([2, 4] * 200), PRIME)) == pytest.approx(np.sqrt(0.0625 - 0.25 / 399))


def brute_force_favorite_share(counts):
    hits = 0.0
    for v in np.flatnonzero(counts):
        others = counts.copy()
        others[v] -= 1
        tied = np.flatnonzero(others == others.max())
        hits += counts[v] * (v in tied) / len(tied)
    return hits / counts.sum()


def test_favorite_share_by_hand():
    assert float(favorite_share(counts_of([42] * 12))) == 1
    # Each 7 matches half the time (the others tie 7 with 8); the 8 never does.
    assert float(favorite_share(counts_of([7, 7, 8]))) == pytest.approx(1 / 3)
    assert float(favorite_share(counts_of([7, 8]))) == 0
    assert float(favorite_share(counts_of([7, 7, 7, 8]))) == pytest.approx(0.75)


def test_favorite_share_matches_leaving_each_answer_out():
    rng = np.random.default_rng(0)
    for _ in range(500):
        counts = rng.multinomial(rng.integers(2, 20), rng.dirichlet(np.full(100, 0.3)))
        assert float(favorite_share(counts)) == pytest.approx(brute_force_favorite_share(counts))


# The sub-scores


def score_all(counts):
    return {name: pytest.approx(float(check(counts))) for name, check in CHECKS.items()}


def test_checks_follow_the_bias_table():
    assert list(CHECKS) == [
        "uniformity", "primes", "multiples_of_10", "multiples_of_5",
        "favorite", "edges", "repeated_digits", "last_digit",
    ]


def test_an_exactly_uniform_run_scores_1_everywhere():
    assert score_all(counts_of(list(range(100)) * 6)) == {name: 1 for name in CHECKS}


def test_the_same_answer_every_time():
    # Gemini 3.1 Flash-Lite in 2.4: 42 on all 12 draws.
    assert score_all(counts_of([42] * 12)) == {
        "uniformity": 0.01,  # a fair pick among 1 value
        "primes": 0.75,  # 25% of answers would have to become prime
        "multiples_of_10": 0.90,
        "multiples_of_5": 0.80,
        "favorite": 0,
        "edges": 0.80,
        "repeated_digits": 0.91,
        "last_digit": 0.10,  # a fair pick among 1 of 10 digits
    }


def test_gpt_5_4_nano_in_2_4():
    # 42 five times, 37, 47 and 73 twice each, 57 once.
    counts = counts_of([42] * 5 + [37, 47, 73] * 2 + [57])
    # 26 of the 132 pairs are equal, so it behaves like a fair pick among 5.1 values.
    assert float(uniformity(counts)) == pytest.approx(132 / 26 / 100)
    # 6 of 12 answers are prime (57 = 3 × 19 isn't). The noise estimate takes
    # 0.5 × 0.5 / 11 off the squared gap of 0.25².
    assert float(primes(counts)) == pytest.approx(1 - np.sqrt(0.25**2 - 0.5 * 0.5 / 11))


def test_scores_stay_in_0_to_1():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 700))
        counts = rng.multinomial(n, rng.dirichlet(np.full(100, rng.choice([0.05, 1, 50]))))
        for check in CHECKS.values():
            assert 0 <= float(check(counts)) <= 1


def test_a_batch_gives_one_score_per_run():
    runs = [counts_of([7, 17, 47, 50, 7]), counts_of([42] * 12)]
    batch = np.stack(runs)
    for check in CHECKS.values():
        assert check(batch).shape == (2,)
        assert list(check(batch)) == [pytest.approx(float(check(r))) for r in runs]


@pytest.mark.parametrize("check", CHECKS.values(), ids=list(CHECKS))
@pytest.mark.parametrize("values", [[], [42]])
def test_rejects_fewer_than_2_answers(check, values):
    with pytest.raises(ValueError):
        check(counts_of(values))


def test_rejects_the_wrong_number_of_counts():
    with pytest.raises(ValueError):
        uniformity(np.ones(99))
