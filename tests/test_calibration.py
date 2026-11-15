import numpy as np
import pytest

from randombench.calibration import (
    MODELS,
    PERFECT,
    calibrate,
    coverage,
    main,
    sample_counts,
    sample_replies,
    true_scores,
)
from randombench.checks import CHECKS, favorite_share, repeat_rate, score_answers


def test_made_up_models_are_distributions():
    for p in MODELS.values():
        assert p.sum() == pytest.approx(1) and (p >= 0).all()


def test_true_scores_follow_score_answers():
    assert list(true_scores(PERFECT)) == list(score_answers([1, 2]))
    assert true_scores(PERFECT, invalid=0.25)["headline"] == pytest.approx(0.75)


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


# D7: from N = 300 up, a model's average score is within 0.02 of its true score


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


# The 95% intervals (4.5)


def test_simulated_replies_have_the_share_of_invalid_ones():
    replies = sample_replies(PERFECT, 10_000, 0.2, np.random.default_rng(4))
    assert len(replies) == 10_000
    assert replies.count(None) / len(replies) == pytest.approx(0.2, abs=0.02)
    assert all(r is None or 0 <= r <= 99 for r in replies)


@pytest.mark.parametrize("name", ["llm-like", "two-tied"])
def test_intervals_hold_the_true_score_about_95_percent_of_the_time(name):
    # Models whose biases stand clear of the noise, as the real models' main
    # biases do. Near a perfect score the cap at 1 skews the coverage either
    # way, which PLAN.md 4.5a records, and a set a model rarely answers gets
    # too narrow an interval (5.4b). 300 runs measure coverage to about ±1.3 points.
    held = coverage(MODELS[name], 624, trials=300, invalid=0.1, n_resamples=500, seed=5)
    for score, share in held.items():
        assert 0.88 <= share <= 0.99, score


def test_one_value_is_always_held_exactly():
    # Every resample is the same, so the interval is ±0, and the score is the truth.
    assert set(coverage(MODELS["one-value"], 100, trials=20, n_resamples=100).values()) == {1.0}


def test_main_prints_every_check(capsys):
    main(["600", "--trials", "200"])
    out = capsys.readouterr().out
    for name in CHECKS:
        assert name in out


def test_main_prints_coverage_for_every_model(capsys):
    main(["--coverage", "100", "--trials", "5"])
    out = capsys.readouterr().out
    assert "headline" in out
    for name in MODELS:
        assert name in out
