"""Describing one batch's answers, offline."""

import json

from randombench.batch import by_count, main, most_common, rundown
from randombench.checks import counts_of

NANO = "openai/gpt-5.4-nano-2026-03-17"


def draw(run_id, index, reply=None, error=None, prompt="plain", model=NANO):
    """A stored record, as extract.py writes it (only the fields the rundown reads matter)."""
    return {"model": model, "run_id": run_id, "task_version": 14, "draw_index": index,
            "prompt_id": f"{prompt}-0123abcd", "reply": reply, "error": error}


def test_groups_values_by_how_often_they_came_up():
    assert by_count(counts_of([42, 42, 42, 7, 7, 73, 99])) == [(3, [42]), (2, [7]), (1, [73, 99])]


def test_most_common_names_every_tied_value():
    assert most_common(counts_of([42, 42, 37, 37, 5])) == "37, 42: 2 of 5 each"
    assert most_common(counts_of([42, 42, 5])) == "42: 2 of 3"
    assert most_common(counts_of([1, 2, 3])) == "none repeats"


def test_counts_failed_calls_and_invalid_replies_apart():
    run = [draw("1", 0, "42"), draw("1", 1, "42"), draw("1", 2, "73"),
           draw("1", 3, "47 or 52"), draw("1", 4, error="APITimeoutError")]
    out = rundown(run, run)
    assert out.startswith("**GPT-5.4 nano, run 1 (v14):** 5 draws, 1 failed call, 1 invalid reply,"
                          " 3 valid answers. They take 2 distinct values, and their mean is 52.3,")
    assert "| 2 | 42 |" in out and "| 1 | 73 |" in out


def test_shows_decades_last_digits_and_prompts():
    run = [draw("1", i, reply, prompt=p) for i, (reply, p) in
           enumerate([("42", "plain"), ("47", "plain"), ("73", "terse"), ("7", "terse"), ("x", "terse")])]
    out = rundown(run, run)
    assert "| 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |" in out  # 7, 42 and 47, 73
    assert "| 0 | 0 | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |" in out  # 42, 73, 7 and 47
    assert "| plain | 2 | 0 | 2 | none repeats |" in out
    assert "| terse | 2 | 1 | 2 | none repeats |" in out


def test_a_first_run_has_nothing_to_compare_with():
    run = [draw("1", i, str(i)) for i in range(10)]
    out = rundown(run, run)
    assert "**Its first run**" in out and "All " not in out


def test_compares_with_all_the_models_runs():
    run = [draw("2", i, "42") for i in range(4)]
    earlier = [draw("1", i, str(i)) for i in range(4)]
    out = rundown(run, earlier + run)
    # One value every time: a fair pick among 1 of the 100 values.
    assert "| This batch | 4 | 0.010 ± 0.000 | 42: 4 of 4 |" in out
    assert "| All 2 runs | 8 | " in out and "| 42: 4 of 8 |" in out


def test_a_batch_whose_calls_all_failed():
    run = [draw("1", i, error="NotFoundError") for i in range(3)]
    assert rundown(run, run).endswith("0 valid answers.\n\nNo valid answers to describe.")


def test_main_reads_the_models_other_runs_from_the_same_data_folder(tmp_path, capsys):
    folder = tmp_path / "gpt-5.4-nano-2026-03-17"
    folder.mkdir()
    other = tmp_path / "gemini" / "9.jsonl"
    other.parent.mkdir()
    for file, records in [(folder / "1.jsonl", [draw("1", i, "42") for i in range(3)]),
                          (folder / "2.jsonl", [draw("2", i, "73") for i in range(3)]),
                          (other, [draw("9", 0, "5", model="google/gemini")])]:
        file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    main([str(folder / "2.jsonl")])
    out = capsys.readouterr().out
    assert "run 2 (v14)" in out
    assert "| All 2 runs | 6 | " in out  # the other model's draw isn't counted
