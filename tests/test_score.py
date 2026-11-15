import json

import pytest

from randombench.checks import CHECKS
from randombench.score import format_table, main, score_draws

NANO = "openai/gpt-5.4-nano-2026-03-17"
GEMINI = "google/gemini-3.1-flash-lite-preview"


def draw(model, run_id, index, reply=None, error=None):
    """A stored record, as extract.py writes it (only the fields scoring reads matter)."""
    return {"model": model, "run_id": run_id, "draw_index": index, "reply": reply, "error": error}


def test_merges_a_models_runs_and_leaves_out_failed_calls():
    draws = [
        draw(NANO, "1", 0, "42"),
        draw(NANO, "1", 1, "**37**"),
        draw(NANO, "1", 2, error="APITimeoutError: Request timed out."),
        draw(NANO, "2", 0, "47 or 52"),  # a second run, with an invalid reply
        draw(NANO, "2", 1, "42"),
        draw(GEMINI, "3", 0, "42"),
        draw(GEMINI, "3", 1, "42"),
    ]
    gemini, nano = score_draws(draws)  # in model name order
    assert (nano.model, nano.draws, nano.failed, nano.invalid, nano.valid) == (NANO, 5, 1, 1, 3)
    # The failed call isn't an invalid answer: 1 of 4 replies was invalid.
    assert nano.scores["invalid_answers"] == pytest.approx(3 / 4)
    # 42, 37, 42: one equal pair of three, a fair pick among 3 values.
    assert nano.scores["uniformity"] == pytest.approx(0.03)
    assert (gemini.valid, gemini.scores["uniformity"]) == (2, pytest.approx(0.01))
    # Each score has an interval. Gemini's resamples are all 42, so its are ±0.
    assert list(nano.intervals) == list(nano.scores)
    assert gemini.intervals["uniformity"] == 0


def test_a_model_with_only_failed_calls_has_no_scores():
    (result,) = score_draws([draw(NANO, "1", 0, error="RateLimitError"), draw(NANO, "1", 1, error="RateLimitError")])
    assert (result.draws, result.failed, result.valid, result.scores, result.intervals) == (2, 2, 0, {}, {})


def test_a_model_with_one_valid_answer_has_no_distribution_scores():
    (result,) = score_draws([draw(NANO, "1", 0, "42"), draw(NANO, "1", 1, "forty-two")])
    assert result.scores["invalid_answers"] == 0.5
    assert all(result.scores[name] is None for name in CHECKS)
    assert all(result.intervals[name] is None for name in CHECKS)
    assert "—" in format_table([result])


def test_main_prints_a_row_per_model(tmp_path, capsys):
    run = tmp_path / "gpt-5.4-nano-2026-03-17" / "1.jsonl"
    run.parent.mkdir()
    run.write_text("".join(json.dumps(draw(NANO, "1", i, str(i))) + "\n" for i in range(10)), encoding="utf-8")
    main(["--data", str(tmp_path)])
    out = capsys.readouterr().out
    assert NANO in out and "uniform" in out
    assert out.count(NANO) == 2  # a row of scores, then a row of intervals
    assert "± 95% interval" in out


def test_main_leaves_out_a_model_whose_calls_all_failed(tmp_path, capsys):
    grok = "xai/grok-4.5-0708"
    for model, records in [
        (NANO, [draw(NANO, "1", i, str(i)) for i in range(10)]),
        (grok, [draw(grok, "2", i, error="NotFoundError") for i in range(3)]),
    ]:
        run = tmp_path / model.split("/")[1] / "1.jsonl"
        run.parent.mkdir()
        run.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    main(["--data", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count(NANO) == 2
    assert out.count(grok) == 1 and "Left out: xai/grok-4.5-0708, whose 3 calls all failed." in out


def test_main_says_so_when_there_are_no_draws(tmp_path, capsys):
    main(["--data", str(tmp_path)])
    assert "no draws" in capsys.readouterr().out
