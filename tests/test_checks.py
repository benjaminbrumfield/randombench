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
    score_answers,
    score_intervals,
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
    # 42 five times; 37, 47, and 73 twice each; 57 once.
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


# All the sub-scores for parsed replies


def test_score_answers_gives_the_headline_then_the_bias_tables_order():
    assert list(score_answers([1, 2])) == ["headline", "invalid_answers", *CHECKS]


def test_score_answers_scores_only_the_valid_answers():
    scores = score_answers([42] * 10 + [None] * 2)
    assert scores["invalid_answers"] == pytest.approx(10 / 12)
    assert scores["uniformity"] == pytest.approx(0.01)  # the Nones don't count as answers
    assert scores["favorite"] == 0
    assert scores["headline"] == pytest.approx(10 / 12 * 0.01)


def test_score_answers_with_no_invalid_replies():
    expected = {"headline": 1, "invalid_answers": 1, **{name: 1 for name in CHECKS}}
    assert score_answers(list(range(100)) * 6) == expected


def test_the_headline_is_the_valid_share_times_uniformity():
    # nano's 12 draws from 2.4, plus 4 invalid replies.
    scores = score_answers([42] * 5 + [37, 47, 73] * 2 + [57] + [None] * 4)
    assert scores["headline"] == pytest.approx(12 / 16 * (132 / 26 / 100))
    # Biases that uniformity already reflects don't lower the headline again.
    assert scores["headline"] == pytest.approx(scores["invalid_answers"] * scores["uniformity"])


@pytest.mark.parametrize("answers", [[42, None, None], [None], [None, None]])
def test_score_answers_gives_no_distribution_scores_or_headline_below_2_valid(answers):
    scores = score_answers(answers)
    assert scores["invalid_answers"] == pytest.approx(1 - answers.count(None) / len(answers))
    assert all(scores[name] is None for name in CHECKS)
    assert scores["headline"] is None


def test_score_answers_needs_a_reply():
    with pytest.raises(ValueError):
        score_answers([])


# 95% intervals (4.5). Their coverage is checked in test_calibration.py.

# nano's 12 draws from 2.4, 50 times over, plus 100 invalid replies.
SOME_INVALID = ([42] * 5 + [37, 47, 73] * 2 + [57]) * 50 + [None] * 100


def test_score_intervals_cover_the_scores_score_answers_gives():
    assert list(score_intervals(SOME_INVALID)) == list(score_answers(SOME_INVALID))


def test_the_same_seed_gives_the_same_intervals():
    assert score_intervals(SOME_INVALID, seed=3) == score_intervals(SOME_INVALID, seed=3)
    assert score_intervals(SOME_INVALID, seed=3) != score_intervals(SOME_INVALID, seed=4)


def test_the_valid_share_interval_is_the_binomial_one():
    # 600 of 700 replies valid: 1.96 × √(p(1 − p)/n) for a share p of n.
    p = 600 / 700
    expected = 1.96 * np.sqrt(p * (1 - p) / 700)
    assert score_intervals(SOME_INVALID)["invalid_answers"] == pytest.approx(expected, rel=0.03)


def test_the_same_answer_every_time_has_no_spread():
    # Gemini 3.1 Flash-Lite in 4.3: every resample is 42 every time.
    assert set(score_intervals([42] * 624).values()) == {0}


def test_four_times_the_answers_halve_the_interval():
    small, large = score_intervals(SOME_INVALID), score_intervals(SOME_INVALID * 4)
    # The scores with any spread: nano never gave an edge, a multiple of 5, or a repeated digit.
    for name in ["headline", "invalid_answers", "uniformity", "primes", "favorite", "last_digit"]:
        assert large[name] == pytest.approx(small[name] / 2, rel=0.1), name


@pytest.mark.parametrize("answers", [[42, None, None], [None], [None, None]])
def test_no_score_means_no_interval(answers):
    intervals = score_intervals(answers)
    assert all(intervals[name] is None for name in [*CHECKS, "headline"])
    assert intervals["invalid_answers"] >= 0


def test_resamples_too_small_to_score_are_left_out():
    # 2 valid answers of 22: most resamples hold fewer than 2.
    intervals = score_intervals([42, 37] + [None] * 20)
    assert all(np.isfinite(h) for h in intervals.values())


def test_score_intervals_needs_a_reply():
    with pytest.raises(ValueError):
        score_intervals([])
