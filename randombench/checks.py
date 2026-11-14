"""Each bias check's sub-score (PLAN.md 3.3, D7).

A check takes the counts of a model's valid answers: 100 counts, where
counts[v] is how often the model answered v. It also takes a batch of them,
shaped (..., 100), and then gives one score for each. A score is in [0, 1]:
about 1 for a perfect generator, lower the more biased the model.

Each score estimates a property of the model's true distribution, and the
estimate is built so that sampling noise cancels on average. A model's
expected score is then its true score at any N, so a model with fewer
valid answers neither gains nor loses by chance (3.3):

- Uniformity and last digit: the effective number of values K, as K/100
  (K/10 for the last digits). A fair pick among K values repeats itself as
  often as the model does. The repeat rate, the chance that two answers are
  equal, has an exactly unbiased estimate: the share of pairs of answers
  that are equal.
- Primes, round numbers, edges and repeated digits: 1 minus the gap
  between the model's share and the ideal share, which is the share of
  answers that would have to change. The gap is the square root of an
  unbiased estimate of its square.
- Favorite: 1 minus how far the favorite's share exceeds 1%, scaled so that
  one value taking every answer scores 0. The share is estimated by leaving
  one out: how often an answer equals the favorite of all the other answers.

score_answers() gives all the sub-scores for a model's parsed replies,
including the invalid-answer score, which a perfect generator never loses.
It takes parsed answers rather than raw replies, so this module needs no
parser and can be copied into a task file on its own (D6).
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


def _validated(counts) -> tuple[np.ndarray, np.ndarray]:
    """Return the counts as an array and the number of answers in each run."""
    counts = np.asarray(counts)
    if counts.shape[-1] != N_VALUES:
        raise ValueError(f"expected {N_VALUES} counts, got {counts.shape[-1]}")
    n = counts.sum(axis=-1)
    if np.any(n < 2):
        raise ValueError("a score needs at least 2 valid answers")
    return counts, n


# The estimates the scores are built from


def _repeat_rate(binned: np.ndarray) -> np.ndarray:
    """The share of pairs of answers in the same bin, for counts over any bins."""
    n = binned.sum(axis=-1)
    return (binned * (binned - 1)).sum(axis=-1) / (n * (n - 1))


def _effective(binned: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 1 / _repeat_rate(binned)


def repeat_rate(counts) -> np.ndarray:
    """An unbiased estimate of the chance that two answers are equal: the share
    of pairs of answers that are. A perfect generator's is 1/100."""
    return _repeat_rate(_validated(counts)[0])


def effective_values(counts) -> np.ndarray:
    """How many values a fair pick would need to repeat itself as often as the
    model: 1 / repeat rate. Infinite when no two answers are equal."""
    return _effective(_validated(counts)[0])


def last_digit_counts(counts) -> np.ndarray:
    """The 10 counts of the answers' last digits, shaped (..., 10)."""
    counts = np.asarray(counts)
    # Value v = 10 * tens + last, so reshaping puts the last digit on the final axis.
    return counts.reshape(*counts.shape[:-1], 10, 10).sum(axis=-2)


def share_gap(counts, mask) -> np.ndarray:
    """How far the share of answers in the mask is from its ideal, mask.mean().

    The observed squared gap is too big on average by the share's sampling
    variance, s(1 - s)/n. Subtracting an unbiased estimate of that variance,
    s(1 - s)/(n - 1), leaves an unbiased estimate of the true squared gap.
    It can come out negative, as it should for a perfect generator about half
    the time, and is floored at 0 before the square root.
    """
    counts, n = _validated(counts)
    s = counts[..., mask].sum(axis=-1) / n
    squared = (s - mask.mean()) ** 2 - s * (1 - s) / (n - 1)
    return np.sqrt(np.maximum(0, squared))


def favorite_share(counts) -> np.ndarray:
    """How often an answer equals the favorite of all the other answers.

    This leave-one-out share estimates, without bias, the chance that a new
    answer equals the model's current favorite: exactly 1/100 for a perfect
    generator, where the plain top share is about 2.2% at N = 600. Ties among
    the other answers' favorites count fractionally.

    Only a value that leads alone can match: take one of its answers away and
    it must still be among the most common. If its lead is exactly one, it
    then ties with every value in second place.
    """
    counts, n = _validated(counts)
    ordered = np.sort(counts, axis=-1)
    top, second = ordered[..., -1], ordered[..., -2]
    tied_after = 1 + np.where(top - 1 == second, (counts == second[..., None]).sum(axis=-1), 0)
    return np.where(top > second, top / n / tied_after, 0.0)


# The sub-scores


def uniformity(counts) -> np.ndarray:
    """The effective number of values, out of 100."""
    return np.minimum(1, effective_values(counts) / N_VALUES)


def last_digit(counts) -> np.ndarray:
    """The effective number of last digits, out of 10."""
    counts, _ = _validated(counts)
    return np.minimum(1, _effective(last_digit_counts(counts)) / 10)


def primes(counts) -> np.ndarray:
    """1 minus the gap from 25% primes."""
    return 1 - share_gap(counts, PRIME)


def multiples_of_10(counts) -> np.ndarray:
    """1 minus the gap from 10% multiples of 10, 0 included."""
    return 1 - share_gap(counts, MULTIPLE_OF_10)


def multiples_of_5(counts) -> np.ndarray:
    """1 minus the gap from 20% multiples of 5, 0 included."""
    return 1 - share_gap(counts, MULTIPLE_OF_5)


def favorite(counts) -> np.ndarray:
    """1 minus the favorite's excess over 1%, scaled so that 100% scores 0."""
    excess = (favorite_share(counts) - 1 / N_VALUES) / (1 - 1 / N_VALUES)
    return np.clip(1 - excess, 0, 1)


def edges(counts) -> np.ndarray:
    """1 minus the gap from 20% of answers in 0-9 or 90-99."""
    return 1 - share_gap(counts, EDGE)


def repeated_digits(counts) -> np.ndarray:
    """1 minus the gap from 9% of answers in 11, 22, ..., 99."""
    return 1 - share_gap(counts, REPEATED_DIGITS)


# The distribution checks, in the order of PLAN.md's bias table.
CHECKS = {
    "uniformity": uniformity,
    "primes": primes,
    "multiples_of_10": multiples_of_10,
    "multiples_of_5": multiples_of_5,
    "favorite": favorite,
    "edges": edges,
    "repeated_digits": repeated_digits,
    "last_digit": last_digit,
}

# The distribution checks need this many valid answers. Below it they give None (3.4).
MIN_VALID = 2


def invalid_answers(answers) -> float:
    """1 minus the share of replies the parser rejected, which are None in answers."""
    if not answers:
        raise ValueError("no replies to score")
    return 1 - sum(a is None for a in answers) / len(answers)


def score_answers(answers) -> dict[str, float | None]:
    """Return every sub-score for a model's parsed replies, in the bias table's order.

    answers holds an int in 0..99 for each valid reply and None for each
    invalid one. Failed calls have no reply, so the caller leaves them out,
    and they don't count as invalid (3.4). The distribution checks score only
    the valid answers, and give None when there are fewer than MIN_VALID.
    """
    valid = [a for a in answers if a is not None]
    scores = {"invalid_answers": invalid_answers(answers)}
    counts = counts_of(valid)
    for name, check in CHECKS.items():
        scores[name] = float(check(counts)) if len(valid) >= MIN_VALID else None
    return scores
