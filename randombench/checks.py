"""Each bias check's raw measurement (PLAN.md 3.2, "Bias checks").

A measurement takes the counts of a model's valid answers: 100 counts, where
counts[v] is how often the model answered v. It also takes a batch of them,
shaped (..., 100), and then gives one value for each. That's how the
calibration harness measures thousands of simulated runs at once.

These are measurements, not scores. A prime share of 0.31 is only that. How
far a measurement may stray depends on how far a perfect generator strays
at the same number of answers (calibration.py), and turning a measurement
into a sub-score in [0, 1] is decided in 3.3 (D7).

Invalid replies aren't counted here. Their rate is measured from the parsed
replies, and a perfect generator never gives one.
"""

import numpy as np

N_VALUES = 100  # the answers 0..99


def _values_where(predicate) -> np.ndarray:
    """A mask over the values 0..99: True where the predicate holds."""
    return np.array([predicate(v) for v in range(N_VALUES)])


PRIME = _values_where(lambda v: v > 1 and all(v % d for d in range(2, v)))
MULTIPLE_OF_10 = _values_where(lambda v: v % 10 == 0)
MULTIPLE_OF_5 = _values_where(lambda v: v % 5 == 0)
EDGE = _values_where(lambda v: v <= 9 or v >= 90)
REPEATED_DIGITS = _values_where(lambda v: v >= 10 and v // 10 == v % 10)


def counts_of(values) -> np.ndarray:
    """Return the 100 counts of a list of valid answers."""
    values = np.asarray(values, dtype=int)
    if values.size and (values.min() < 0 or values.max() >= N_VALUES):
        raise ValueError("answers must be in 0..99")
    return np.bincount(values, minlength=N_VALUES)


def _shares(counts) -> np.ndarray:
    """Each value's share of the answers, shaped like counts."""
    counts = np.asarray(counts)
    if counts.shape[-1] != N_VALUES:
        raise ValueError(f"expected {N_VALUES} counts, got {counts.shape[-1]}")
    n = counts.sum(axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("no valid answers to measure")
    return counts / n


def _distance_from_uniform(shares) -> np.ndarray:
    """Total variation distance: half the sum of the gaps between the observed
    and the ideal shares. It's the share of answers that would have to change
    to make the distribution uniform: 0 when it is, nearly 1 when one answer
    takes everything."""
    ideal = 1 / shares.shape[-1]
    return 0.5 * np.abs(shares - ideal).sum(axis=-1)


def uniformity(counts) -> np.ndarray:
    """How far the whole distribution is from uniform, as a total variation distance."""
    return _distance_from_uniform(_shares(counts))


def primes(counts) -> np.ndarray:
    """The share of answers that are prime: 25 of the 100 values."""
    return _shares(counts)[..., PRIME].sum(axis=-1)


def multiples_of_10(counts) -> np.ndarray:
    """The share of answers that are multiples of 10, 0 included: 10 values."""
    return _shares(counts)[..., MULTIPLE_OF_10].sum(axis=-1)


def multiples_of_5(counts) -> np.ndarray:
    """The share of answers that are multiples of 5, 0 included: 20 values."""
    return _shares(counts)[..., MULTIPLE_OF_5].sum(axis=-1)


def favorite(counts) -> np.ndarray:
    """The share of answers taken by the single most common value."""
    return _shares(counts).max(axis=-1)


def edges(counts) -> np.ndarray:
    """The share of answers in 0-9 or 90-99: 20 values."""
    return _shares(counts)[..., EDGE].sum(axis=-1)


def repeated_digits(counts) -> np.ndarray:
    """The share of answers 11, 22, ..., 99: 9 values."""
    return _shares(counts)[..., REPEATED_DIGITS].sum(axis=-1)


def last_digit(counts) -> np.ndarray:
    """How far the last digits' shares are from 10% each, as a total variation distance."""
    shares = _shares(counts)
    # Value v = 10 * tens + last, so reshaping puts the last digit on the final axis.
    by_last_digit = shares.reshape(*shares.shape[:-1], 10, 10).sum(axis=-2)
    return _distance_from_uniform(by_last_digit)


# In the order of PLAN.md's bias table, after the invalid-answer rate.
MEASUREMENTS = {
    "uniformity": uniformity,
    "primes": primes,
    "multiples_of_10": multiples_of_10,
    "multiples_of_5": multiples_of_5,
    "favorite": favorite,
    "edges": edges,
    "repeated_digits": repeated_digits,
    "last_digit": last_digit,
}
