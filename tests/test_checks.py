import numpy as np
import pytest

from randombench.checks import (
    EDGE,
    MEASUREMENTS,
    MULTIPLE_OF_5,
    MULTIPLE_OF_10,
    PRIME,
    REPEATED_DIGITS,
    counts_of,
    edges,
    favorite,
    last_digit,
    multiples_of_5,
    multiples_of_10,
    primes,
    repeated_digits,
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


# Measurements


def measure_all(counts):
    return {name: pytest.approx(float(m(counts))) for name, m in MEASUREMENTS.items()}


def test_measurements_follow_the_bias_table():
    assert list(MEASUREMENTS) == [
        "uniformity", "primes", "multiples_of_10", "multiples_of_5",
        "favorite", "edges", "repeated_digits", "last_digit",
    ]


def test_an_exactly_uniform_run_measures_the_ideal():
    counts = counts_of(list(range(100)) * 6)
    assert measure_all(counts) == {
        "uniformity": 0, "primes": 0.25, "multiples_of_10": 0.10, "multiples_of_5": 0.20,
        "favorite": 0.01, "edges": 0.20, "repeated_digits": 0.09, "last_digit": 0,
    }


def test_the_same_answer_every_time():
    # Gemini 3.1 Flash-Lite in 2.4: 42 on all 12 draws.
    counts = counts_of([42] * 12)
    assert measure_all(counts) == {
        "uniformity": 0.99, "primes": 0, "multiples_of_10": 0, "multiples_of_5": 0,
        "favorite": 1, "edges": 0, "repeated_digits": 0, "last_digit": 0.9,
    }


def test_a_hand_counted_run():
    counts = counts_of([7, 17, 47, 50])
    assert measure_all(counts) == {
        # Four values at 25% each, against 1%: half of (4 × 0.24 + 96 × 0.01).
        "uniformity": 0.96,
        "primes": 0.75,
        "multiples_of_10": 0.25,
        "multiples_of_5": 0.25,
        "favorite": 0.25,
        "edges": 0.25,
        "repeated_digits": 0,
        # Last digits 7, 7, 7, 0 against 10% each: half of (0.65 + 0.15 + 8 × 0.1).
        "last_digit": 0.8,
    }


def test_a_repeated_digit_and_an_edge():
    counts = counts_of([99, 0, 55, 10])
    assert float(repeated_digits(counts)) == pytest.approx(0.5)
    assert float(edges(counts)) == pytest.approx(0.5)
    assert float(multiples_of_10(counts)) == pytest.approx(0.5)
    assert float(multiples_of_5(counts)) == pytest.approx(0.75)


def test_a_batch_gives_one_value_per_run():
    runs = [counts_of([7, 17, 47, 50]), counts_of([42] * 12)]
    batch = np.stack(runs)
    for measure in MEASUREMENTS.values():
        assert measure(batch).shape == (2,)
        assert list(measure(batch)) == [pytest.approx(float(measure(r))) for r in runs]


@pytest.mark.parametrize("measure", MEASUREMENTS.values(), ids=list(MEASUREMENTS))
def test_rejects_a_run_with_no_valid_answers(measure):
    with pytest.raises(ValueError):
        measure(counts_of([]))


def test_rejects_the_wrong_number_of_counts():
    with pytest.raises(ValueError):
        uniformity(np.ones(99))


def test_measurements_are_unaffected_by_scale():
    small = counts_of([7, 17, 47, 50])
    for measure in (uniformity, primes, favorite, last_digit, multiples_of_5):
        assert float(measure(small * 10)) == pytest.approx(float(measure(small)))
